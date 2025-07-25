import json
from PIL import Image
from torch.utils.data import Dataset

from src.data.datasets import FHMDataset

import torchvision.transforms as T


root_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/fhmd'
img_size = 256

val_transform_img = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomCrop(int(img_size*0.94)),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_dataset = FHMDataset(f'{root_path}/dev.jsonl', f'{root_path}/imgs', img_transform=val_transform_img, text_transform=None)

img, text, label = val_dataset[0]

# -- Jeśli img jest torch.Tensor, przekonwertuj na PIL.Image --
if isinstance(img, T.functional.Tensor):
    pil_img = T.ToPILImage()(img)
else:
    pil_img = img

# 1. Zapisz obraz na dysk
pil_img.save("data/images/przykład_z_danych_fhmd.jpg")

# 2. Wypisz czytelnie pozostałe informacje
print("**Obraz zapisany jako:**", "przykład_z_danych.jpg")
print("**Tekst:**", text)
print("**Etykieta (label):**", label)

