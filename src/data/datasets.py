
import json
import pickle
import random
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class DatasetClassRemapped(Dataset):
    def __init__(self, base_dataset, class_mapping):
        self.base_dataset = base_dataset
        self.class_mapping = class_mapping
        self.transform = getattr(base_dataset, "transform", None)
        self.targets = base_dataset.targets

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.class_mapping[label]

    def __len__(self):
        return len(self.base_dataset)
    
    
class RemappedSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, class_mapping, transform=None):
        """
        :param base_dataset: oryginalny dataset (np. ImageFolder, OxfordIIITPet)
        :param indices: lista indeksów (podzbiór)
        :param class_mapping: słownik mapujący oryginalne etykiety -> nowe etykiety
        :param transform: opcjonalna transformacja obrazu
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.transform = transform or getattr(base_dataset, 'transform', None)
        self.targets = [class_mapping[label] for label in np.array(base_dataset.targets)[np.array(indices)]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        label = self.class_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label


    
class PairDataset(Dataset):
    def __init__(self, base_dataset, class_indices, pairs_file=None, transform1=None, transform2=None):
        """
        base_dataset: dataset (np. CIFAR10)
        class_indices: lista wybranych klas (np. [0, 1, 2])
        pairs_file: opcjonalny path do pliku z gotowymi parami
        transform1: transformacje do zastosowania na obrazach lewych
        transform2: transformacje do zastosowania na obrazach prawych
        """
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.class_indices = class_indices

        # wybierz indeksy próbek należących do każdej z wybranych klas
        self.class_to_indices = {c: [] for c in class_indices}
        for idx, (_, label) in enumerate(base_dataset):
            if label in class_indices:
                self.class_to_indices[label].append(idx)
        
        # jeśli podano plik z parami, załaduj je
        if pairs_file and os.path.exists(pairs_file):
            print(f"Loading pairs from {pairs_file}")
            with open(pairs_file, 'rb') as f:
                self.pairs = pickle.load(f)
        else: # jeśli nie ma pliku z parami, wygeneruj je
            self.pairs = []
            for c in class_indices:
                indices = self.class_to_indices[c][:]
                random.shuffle(indices)
                # Jeśli liczba próbek nieparzysta, odrzuć jedną
                if len(indices) % 2 != 0:
                    indices = indices[:-1]
                for i in range(0, len(indices), 2):
                    self.pairs.append( (indices[i], indices[i+1], c) ) # (indeks1, indeks2, klasa)
            if pairs_file:
                with open(pairs_file, 'wb') as f:
                    pickle.dump(self.pairs, f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        if self.transform1 is not None:
            img1 = self.transform1(img1)
        if self.transform2 is not None:
            img2 = self.transform2(img2)
        # Przesuń etykiety do zakresu 0-2
        class_map = {c: i for i, c in enumerate(self.class_indices)}
        label = class_map[label]
        return img1, img2, label


class FHMDataset(Dataset):
    def __init__(self, jsonl_path, img_dir, img_transform=None, text_transform=None):
        """
        jsonl_path: ścieżka do pliku .jsonl (np. 'train.jsonl')
        img_dir: folder z obrazami (np. 'img')
        transform: opcjonalna transformacja dla obrazu (np. torchvision.transforms)
        text_transform: opcjonalna transformacja dla tekstu (np. tokenizer)
        """
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = f"{self.img_dir}/{sample['img'].split('/')[-1]}"
        img = Image.open(img_path).convert('RGB')
        text = sample['text']
        label = int(sample['label'])

        if self.img_transform:
            img = self.img_transform(img)
        if self.text_transform:
            text = self.text_transform(text) if random.random() < 0.5 else text

        return img, text, label