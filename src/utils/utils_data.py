import numpy as np
import pickle
import random
import logging

from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10




def prepare_loaders(data_params):
    from src.utils.mapping_new import DATASET_NAME_MAP
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        data_params (dict): 
            Dictionary containing all necessary dataset and DataLoader parameters. 
            Must include:
                - 'dataset_name': str, key for DATASET_NAME_MAP
                - 'dataset_params': dict, parameters for the dataset constructor
                - 'loader_params': dict, parameters for DataLoader (e.g., batch_size, num_workers)

    Returns:
        dict: 
            Dictionary with DataLoaders for each phase:
                - 'train': DataLoader for training set (shuffled)
                - 'val': DataLoader for validation set (not shuffled)
                - 'test': DataLoader for test set (not shuffled)
    """
    # train_dataset, test_dataset = DATASET_NAME_MAP[data_params['dataset_name']](**data_params['dataset_params'])
    train_dataset, val_dataset, test_dataset = DATASET_NAME_MAP[data_params['dataset_name']](**data_params['dataset_params'])
    
    loaders = {
        'train': DataLoader(train_dataset, shuffle=True, **data_params['loader_params']),
        'val': DataLoader(val_dataset, shuffle=False, **data_params['loader_params']),
        'test': DataLoader(test_dataset, shuffle=False, **data_params['loader_params'])
    }
    
    return loaders


import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

def mesh_deepcopy(mesh):
    import copy
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    if mesh.has_vertex_normals():
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    if mesh.has_vertex_colors():
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    if mesh.has_triangle_uvs():
        new_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(mesh.triangle_uvs))
    return new_mesh

def render_two_views_for_model(off_path, out_dir, img_size=(224, 224)):
    mesh = o3d.io.read_triangle_mesh(off_path)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()

    angles = [0, 180]  # front, rear
    view_names = ['front', 'rear']

    for i, (angle, vname) in enumerate(zip(angles, view_names)):
        mesh_copy = mesh_deepcopy(mesh)
        rad = np.deg2rad(angle)
        R = mesh_copy.get_rotation_matrix_from_axis_angle([0, rad, 0])
        mesh_copy.rotate(R, center=mesh_copy.get_center())
        vis.clear_geometries()
        vis.add_geometry(mesh_copy)
        vis.poll_events()
        vis.update_renderer()
        img_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(off_path))[0]}_{vname}.png")
        vis.capture_screen_image(img_path, do_render=True)
    vis.destroy_window()

def render_all_off_in_class(class_dir, out_dir):
    """Renderuje wszystkie pliki .off w danej klasie."""
    os.makedirs(out_dir, exist_ok=True)
    for split in ['train', 'test']:
        split_in = os.path.join(class_dir, split)
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        for off_file in tqdm([f for f in os.listdir(split_in) if f.endswith('.off')], desc=f"{class_dir}/{split}"):
            model_in = os.path.join(split_in, off_file)
            model_out = split_out
            render_two_views_for_model(model_in, model_out)

def render_modelnet40_all_classes(modelnet_root, out_root):
    """Renderuje wszystkie klasy ModelNet40."""
    classes = [d for d in os.listdir(modelnet_root) if os.path.isdir(os.path.join(modelnet_root, d))]
    for cls in tqdm(classes, desc="Klasy"):
        class_in = os.path.join(modelnet_root, cls)
        class_out = os.path.join(out_root, cls)
        render_all_off_in_class(class_in, class_out)

def fix_off_header(file_path):
    with open(file_path, 'r') as f:
        first = f.readline()
        if not first.strip() == 'OFF':
            # Plik jest uszkodzony
            numbers = first.strip()[3:]  # Wytnij 'OFF'
            rest = f.read()
            with open(file_path, 'w') as wf:
                wf.write('OFF\n')
                wf.write(numbers + '\n')
                wf.write(rest)

# Przykład wywołania na folderze:
mn="/net/pr2/projects/plgrid/plggdnnp/datasets/ModelNet40"
mn2d="/net/pr2/projects/plgrid/plggdnnp/datasets/ModelNet40_2D"
# import glob
# for f in glob.glob(f'{mn}/*/*/*.off'):
#     fix_off_header(f)
    

render_modelnet40_all_classes(mn, mn2d)