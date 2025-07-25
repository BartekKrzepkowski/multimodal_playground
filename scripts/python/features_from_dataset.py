import sys

import torch
import torchvision.transforms as T
from torchvision.models import resnet18
import h5py
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel

class MMIMDbNumpyDataset(Dataset):
   def __init__(self, images_path, text_label_path, transform=None):
       # Ładuj obrazy przez mmap (oszczędnie)
       self.images = np.load(images_path, mmap_mode='r')
       # Ładuj metadane (data.npy)
       self.data = np.load(text_label_path, allow_pickle=True)
       self.transform = transform


       # Sprawdź orientacyjnie shape
       print("Obrazy shape:", self.images.shape)
       print("Pierwszy rekord data:", self.data[0])


   def __len__(self):
       return self.images.shape[0]


   def __getitem__(self, idx):
       img = self.images[idx]


       # Obsłuż różne układy obrazów: (3,H,W) lub (H,W,3)
       if img.shape[0] == 3:
           img_tensor = torch.from_numpy(img).float() / 255.0
       elif img.shape[-1] == 3:
           img = np.transpose(img, (2,0,1))
           img_tensor = torch.from_numpy(img).float() / 255.0
       else:
           raise ValueError(f"Nieoczekiwany shape obrazu: {img.shape}")


       if self.transform:
           img_tensor = self.transform(img_tensor)


       # print(f"Przetwarzanie obrazu {idx}: shape {img_tensor.shape}")


       # Załóżmy, że data[i] to dict: {'genres': ..., 'description': ...}
       meta = self.data[idx]
       genres = np.array(meta['genres'] if isinstance(meta, dict) else meta[1], dtype=np.float32)
       description = meta['description'] if isinstance(meta, dict) else meta[2]
       # Możesz też zwracać inne pola wedle potrzeb


       return img_tensor, genres, description


# Ścieżki i ustawienia
batch_size = 32
img_size = (96, 96)  # Rozmiar obrazów po przeskalowaniu


root_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb'
images_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb/images.npy'
text_label_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb/data.npy'


images_output_npy = f'{root_path}/image_features_resnet18.npy'
text_output_npy = f'{root_path}/text_features_distilbert.npy'
labels_output_npy = f'{root_path}/labels_multi_hot.npy'


transform = T.Compose([
    T.ToPILImage(),
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    dataset = MMIMDbNumpyDataset(
        images_path=images_path,
        text_label_path=text_label_path,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    MAX_LENGTH = 0
    for img, genres, desc in loader:
        curr_len = len(desc)
        if MAX_LENGTH < curr_len:
            MAX_LENGTH = curr_len

    print(MAX_LENGTH)
    
    # ---- 2. Przygotowanie modeli ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # ResNet18 bez FC
    img_model = resnet18(pretrained=True)
    img_model.fc = torch.nn.Identity()
    img_model.eval().to(device)


    # DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    text_model.eval()

    # ---- 3. Ekstrakcja cech batchowo ----


    img_feats_list = []
    txt_feats_list = []
    labels_list = []


    with torch.no_grad():
        for imgs_batch, labels, descs in tqdm(loader):
            # ==== 1. Obrazy ====
            imgs_batch = imgs_batch.to(device)
            img_feats = img_model(imgs_batch).cpu().numpy()  # (B, 512)
            img_feats_list.append(img_feats)


            # ==== 2. Teksty ====
            # Tokenizacja batchowa
            enc = tokenizer(list(descs), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k,v in enc.items()}
            out = text_model(**enc)
            txt_feats = out.last_hidden_state[:,0,:].cpu().numpy()  # (B, 768)
            txt_feats_list.append(txt_feats)

            labels_list.append(labels)

        # ---- 4. Zapis do plików ----
        img_feats_arr = np.concatenate(img_feats_list, axis=0)
        txt_feats_arr = np.concatenate(txt_feats_list, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)
        np.save(images_output_npy, img_feats_arr)
        np.save(text_output_npy, txt_feats_arr)
        np.save(labels_output_npy, labels_arr)


        print("Ekstrakcja embeddingów zakończona!")
    

if __name__ == "__main__":
    main()
