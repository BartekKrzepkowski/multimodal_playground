import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import Counter

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
val_dataset = FHMDataset(f'{root_path}/train.jsonl', f'{root_path}/imgs', img_transform=val_transform_img, text_transform=None)

img, text, label = val_dataset[0]

if isinstance(img, T.functional.Tensor):
    pil_img = T.ToPILImage()(img)
else:
    pil_img = img

# 1. Zapisz obraz na dysk
pil_img.save("data/images/przykład_z_danych_fhmd.jpg")
print("**Obraz zapisany jako:**", "przykład_z_danych.jpg")
print("**Tekst:**", text)
print("**Etykieta (label):**", label)

data_params = {
    'dataset_name' : 'fhmd',
    'dataset_params': {},
    'loader_params': {'batch_size': 125, 'pin_memory': True, 'num_workers': 12}
}

loader = DataLoader(val_dataset, shuffle=False, **data_params['loader_params'])

from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lengths = []
all_labels = []

for img, text, label in tqdm(loader):
    # text jest listą/torch.Tensor tekstów w batchu
    if isinstance(text, str):
        texts = [text]
    elif not isinstance(text, list):
        texts = list(text)
    else:
        texts = text

    # Dodaj batchowe etykiety do listy
    # (jeśli label to tensor: .tolist(), jeśli lista: extend)
    if hasattr(label, 'tolist'):
        batch_labels = label.tolist()
    else:
        batch_labels = list(label)
    all_labels.extend(batch_labels)

    for t in texts:
        if not isinstance(t, str) or t.strip() == "":
            print("Pomijam t:", t, type(t))
            continue
        tokens = tokenizer.encode(t, add_special_tokens=True)
        lengths.append(len(tokens))

# ---- Statystyki długości tekstów ----
lengths = np.array(lengths)
print("\n===== Statystyki długości tekstów =====")
print(f"Liczba tekstów:           {len(lengths)}")
print(f"Średnia długość:          {np.mean(lengths):.2f}")
print(f"Mediana długości:         {np.median(lengths)}")
print(f"Min długość:              {np.min(lengths)}")
print(f"Maksymalna długość:       {np.max(lengths)}")
print(f"90 percentyl:             {np.percentile(lengths, 90)}")
print(f"95 percentyl:             {np.percentile(lengths, 95)}")
print(f"99 percentyl:             {np.percentile(lengths, 99)}")

for perc in [90, 95, 99, 100]:
    print(f"{perc}% tekstów zmieści się w max_length = {int(np.percentile(lengths, perc))}")

# ---- Statystyki rozkładu klas ----
print("\n===== Rozkład klas =====")
labels_counter = Counter(all_labels)
total = sum(labels_counter.values())
for k in sorted(labels_counter):
    print(f"Klasa {k}: {labels_counter[k]} przykładów ({labels_counter[k]/total*100:.2f}%)")
print(f"Unikalne etykiety: {list(labels_counter.keys())}")
print(f"Liczba przykładów w zbiorze: {total}")

# ---- Dodatkowe statystyki ----
print("\n===== Dodatkowe statystyki =====")
print(f"Liczba unikalnych tekstów: {len(set(lengths))}")
print(f"Najczęstsza długość tekstu: {Counter(lengths).most_common(1)[0][0]} (występuje {Counter(lengths).most_common(1)[0][1]} razy)")
print(f"Czy rozkład klas jest zbalansowany? {'TAK' if max(labels_counter.values())-min(labels_counter.values()) <= 1 else 'NIE'}")

