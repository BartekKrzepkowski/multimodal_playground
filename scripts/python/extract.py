# extract_images_to_npy.py

import numpy as np
import sys

def main(npz_path, npy_path):
    print(f"Wczytywanie archiwum: {npz_path}")
    with np.load(npz_path) as archive:
        print("Dostępne klucze:", archive.files)
        arr = archive['images']
        print("Kształt tablicy:", arr.shape)
        print(f"Zapisywanie do {npy_path} ...")
        np.save(npy_path, arr)
    print("Zakończono.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python extract_images_to_npy.py images.npz images.npy")
        sys.exit(1)
    npz_path = sys.argv[1]
    npy_path = sys.argv[2]
    main(npz_path, npy_path)
