import numpy as np
import matplotlib.pyplot as plt
import os

# ====== Ustawienia ======
images_path = '/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb/images.npy'
output_path = 'data/images/mmimdb_grid.png'
rows, cols = 5, 8   # liczba wierszy i kolumn (zmień wg uznania)
n_images = rows * cols
dpi = 150           # rozdzielczość pliku wynikowego

# ====== Wczytaj obrazy ======
images = np.load(images_path, mmap_mode='r')

# ====== Tworzenie siatki ======
fig, axs = plt.subplots(rows, cols, figsize=(cols*1.8, rows*2))
for i in range(n_images):
    img = images[i]
    # Popraw orientację (3, H, W) -> (H, W, 3)
    if img.shape[0] == 3:
        img_disp = np.transpose(img, (1,2,0))
    else:
        img_disp = img
    # Upewnij się, że uint8
    if img_disp.dtype != np.uint8:
        img_disp = (img_disp * 255).clip(0,255).astype(np.uint8)
    row, col = divmod(i, cols)
    axs[row, col].imshow(img_disp)
    axs[row, col].axis('off')
    axs[row, col].set_title(str(i), fontsize=8)

# Wyłącz puste kratki, jeśli niepełna siatka
for j in range(n_images, rows*cols):
    row, col = divmod(j, cols)
    axs[row, col].axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=dpi)
plt.close()
print(f"Siatka zapisana do pliku: {os.path.abspath(output_path)}")
