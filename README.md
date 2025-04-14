# Thinning-skeleton-hitmiss

from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import erosion, thin, skeletonize
from skimage.util import invert

# Upload gambar
uploaded = files.upload()

# Ambil dan konversi gambar ke grayscale
for fname in uploaded:
    img = Image.open(fname).convert('L')
    gray_image = np.array(img)
    break

# Konversi ke gambar biner (nilai 0 dan 1)
_, binary = cv2.threshold(gray_image, 127, 1, cv2.THRESH_BINARY)

# Structuring Element
se_fg = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=bool)

se_bg = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=bool)

# Fungsi Hit-or-Miss
def hit_or_miss(image, se_fg, se_bg):
    img_comp = invert(image)
    erode_fg = erosion(image, se_fg)
    erode_bg = erosion(img_comp, se_bg)
    return erode_fg & erode_bg

# Proses Hit-or-Miss
binary_bool = binary.astype(bool)
hitmiss_result = hit_or_miss(binary_bool, se_fg, se_bg)

# Proses Thinning dan Skeletonization
thinned = thin(binary_bool)
skeleton = skeletonize(binary_bool)

# Visualisasi hasil
fig, axs = plt.subplots(1, 5, figsize=(18, 4))
axs[0].imshow(gray_image, cmap='gray')
axs[0].set_title("Grayscale")

axs[1].imshow(binary, cmap='gray')
axs[1].set_title("Biner")

axs[2].imshow(hitmiss_result, cmap='gray')
axs[2].set_title("Hit-or-Miss")

axs[3].imshow(thinned, cmap='gray')
axs[3].set_title("Thinning")

axs[4].imshow(skeleton, cmap='gray')
axs[4].set_title("Skeletonization")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
