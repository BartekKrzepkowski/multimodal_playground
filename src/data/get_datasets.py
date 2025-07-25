import random
import os

import torchvision.transforms as T
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data import Subset
from tqdm import tqdm

from src.data.datasets import FHMDataset


def get_cifar100(dataset_path=None, **kwargs):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR100_PATH']
    
    # transformacje
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    normalize = T.Normalize(mean, std)
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.1),
    ])
    transform_eval = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Wczytanie zbioru CIFAR-100
    train_dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform_eval)

    # podział na zbiór walidacyjny i testowy

    test_labels = test_dataset.targets
    test_ratio = 0.4

    indices = list(range(len(test_dataset)))

    # Stratified split
    val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=test_labels,       # KLUCZOWE!
        random_state=42
    )

    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset



def cifar100_pair_dataset(dataset_path=None, class_indices=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR100_PATH']

    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    normalize = T.Normalize(mean, std)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.1),
    ])
    transform_eval = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Wczytanie zbioru CIFAR-100
    train_dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True)
    test_dataset = datasets.CIFAR100(root=dataset_path, train=False, download=True)

    # podział na zbiór walidacyjny i testowy

    test_labels = test_dataset.targets
    test_ratio = 0.4

    indices = list(range(len(test_dataset)))

    # Stratified split
    val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=test_labels,       # KLUCZOWE!
        random_state=42
    )

    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)


    pair_dataset_train = PairDataset(train_dataset, class_indices=class_indices, transform1=transform_train, transform2=transform_train)
    pair_dataset_val = PairDataset(val_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)
    pair_dataset_test = PairDataset(test_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)

    return pair_dataset_train, pair_dataset_val, pair_dataset_test



def cifar10_pair_dataset(dataset_path=None, class_indices=None):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']

    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    normalize = T.Normalize(mean, std)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.1),
    ])
    transform_eval = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Wczytanie zbioru CIFAR-10
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True)

    # podział na zbiór walidacyjny i testowy

    test_labels = test_dataset.targets
    test_ratio = 0.4

    indices = list(range(len(test_dataset)))

    # Stratified split
    val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=test_labels,       # KLUCZOWE!
        random_state=42
    )

    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)


    pair_dataset_train = PairDataset(train_dataset, class_indices=class_indices, transform1=transform_train, transform2=transform_train)
    pair_dataset_val = PairDataset(val_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)
    pair_dataset_test = PairDataset(test_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)

    return pair_dataset_train, pair_dataset_val, pair_dataset_test



def imagenette_pair_dataset(dataset_path=None, class_indices=None, img_height=32, img_width=32):
    PER_CLASS = 1000
    dataset_path = dataset_path if dataset_path is not None else os.environ['IMAGENETTE160_PATH']

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean, std)
    transform_train = T.Compose([
        T.Resize((img_height, img_width)),
        T.RandomCrop((img_height, img_width), padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.1),
    ])
    transform_eval = T.Compose([
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Wczytanie zbioru CIFAR-100
    
    full_dataset = datasets.ImageFolder(dataset_path)
    
    # Wyznacz liczby klas
    class_to_indices = {}
    for idx, (_, label) in enumerate(full_dataset.imgs):
        class_to_indices.setdefault(label, []).append(idx)
    # Upewnij się, że są posegregowane po etykiecie
    selected_classes = class_indices if class_indices is not None else list(class_to_indices.keys())

    # Zbierz po per_class z każdej klasy
    chosen_indices = []
    for c in selected_classes:
        idxs = class_to_indices[c]
        if len(idxs) < PER_CLASS:
            raise ValueError(f"Za mało obrazów w klasie {c}, jest {len(idxs)}, potrzebne {PER_CLASS}")
        chosen = random.sample(idxs, PER_CLASS)
        chosen_indices.extend(chosen)

    # Wyodrębnij podzbiór z pełnego datasetu
    subset_dataset = Subset(full_dataset, chosen_indices)
    # Stwórz listę labeli tego podzbioru (w tej samej kolejności!)
    subset_labels = [full_dataset.imgs[i][1] for i in chosen_indices]
    subset_labels = np.array(subset_labels)

    train_ratio = 0.6

    val_ratio = 0.2

    # Dzielimy na train, val, test stratified
    train_size = int(train_ratio * PER_CLASS)
    val_size = int(val_ratio * PER_CLASS)
    test_size = PER_CLASS - train_size - val_size
    assert train_size + val_size + test_size == PER_CLASS

    train_indices, val_indices, test_indices = [], [], []
    for c in selected_classes:
        # Pobierz indeksy klasy w subset_dataset (czyli 0..len(subset)-1)
        class_mask = np.where(subset_labels == c)[0]
        np.random.shuffle(class_mask)
        train_indices.extend(class_mask[:train_size])
        val_indices.extend(class_mask[train_size:train_size+val_size])
        test_indices.extend(class_mask[train_size+val_size:train_size+val_size+test_size])

    # Tworzymy Subsety
    train_set = Subset(subset_dataset, train_indices)
    val_set = Subset(subset_dataset, val_indices)
    test_set = Subset(subset_dataset, test_indices)


    # pair_dataset_train = PairDataset(full_datasettrain_dataset, class_indices=class_indices, transform1=transform_train, transform2=transform_train)
    # pair_dataset_val = PairDataset(val_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)
    # pair_dataset_test = PairDataset(test_dataset, class_indices=class_indices, transform1=transform_eval, transform2=transform_eval)

    # return pair_dataset_train, pair_dataset_val, pair_dataset_test


    pass  # TODO: Implement PairDataset for imagenette


def get_mmimdb_pair_dataset():
    """
    Wczytuje pary obrazów z MM-IMDb.
    :param dataset_path: ścieżka do folderu z danymi
    :param class_indices: lista klas do uwzględnienia
    :return: PairDataset dla MM-IMDb
    """
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    import torch
    from torch.utils.data import TensorDataset

    root_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb'
    img_feats_path = f'{root_path}/image_features_resnet18.npy'
    txt_feats_path = f'{root_path}/text_features_distilbert.npy'
    labels_path = f'{root_path}/labels_multi_hot.npy'

    # Załaduj multi-hot etykiety
    # Załaduj dane
    img_feats = np.load(img_feats_path)
    txt_feats = np.load(txt_feats_path)
    labels = np.load(labels_path)  # shape (N, num_labels)

    print(f"Załadowano dane o w rozmiarze: {img_feats.shape}, {txt_feats.shape}, {labels.shape}")

    # Split: train (80%), val (10%), test (10%)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(mskf.split(np.zeros(len(labels)), labels))

    # Indeksy:
    train_idx, val_idx = splits[0]
    val_idx, test_idx = np.array_split(val_idx, 2)

    # Sprawdź rozkład klas:
    print("train:", np.sum(labels[train_idx], axis=0))
    print("val:", np.sum(labels[val_idx], axis=0))
    print("test:", np.sum(labels[test_idx], axis=0))

    labels = np.load(labels_path)

    # Upewnij się, że podział idxów masz jak wyżej:
    train_img = torch.tensor(img_feats[train_idx]).float()
    train_txt = torch.tensor(txt_feats[train_idx]).float()
    train_lab = torch.tensor(labels[train_idx]).float()
    val_img = torch.tensor(img_feats[val_idx]).float()
    val_txt = torch.tensor(txt_feats[val_idx]).float()
    val_lab = torch.tensor(labels[val_idx]).float()
    test_img = torch.tensor(img_feats[test_idx]).float()
    test_txt = torch.tensor(txt_feats[test_idx]).float()
    test_lab = torch.tensor(labels[test_idx]).float()

    print(f"train_img: {train_img.shape}, train_txt: {train_txt.shape}, train_lab: {train_lab.shape}")
    print(f"val_img: {val_img.shape}, val_txt: {val_txt.shape}, val_lab: {val_lab.shape}")
    print(f"test_img: {test_img.shape}, test_txt: {test_txt.shape}, test_lab: {test_lab.shape}")

    # Datasety PyTorch:
    train_dataset = TensorDataset(train_img, train_txt, train_lab)
    val_dataset = TensorDataset(val_img, val_txt, val_lab)
    test_dataset = TensorDataset(test_img, test_txt, test_lab)

    return train_dataset, val_dataset, test_dataset


def get_fhmd(img_size=256):
    from transformers import pipeline
    # import nlpaug.augmenter.word as naw
    root_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/fhmd'

    train_transform_text = None
    # train_transform_text = naw.SynonymAug(aug_src='wordnet')
    # train_transform_text = lambda text: pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")[text][0]['generated_text']

    train_transform_img = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomResizedCrop(int(img_size*0.94), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform_img = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(int(img_size*0.94)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = FHMDataset(f'{root_path}/train.jsonl', f'{root_path}/imgs', img_transform=train_transform_img, text_transform=train_transform_text)
    val_dataset = FHMDataset(f'{root_path}/dev.jsonl', f'{root_path}/imgs', img_transform=val_transform_img, text_transform=None)
    test_dataset = FHMDataset(f'{root_path}/test.jsonl', f'{root_path}/imgs', img_transform=val_transform_img, text_transform=None) # to i tak nie ma sensu, bo nie ma etykiet

    return train_dataset, val_dataset, test_dataset