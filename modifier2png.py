import numpy as np
from PIL import Image

modifier = np.load("result/06_21_17_38_scaled_modifier.npy")

modifier = 255 - modifier * 255
img = np.clip(modifier, 0, 255).astype(np.uint8)
img = np.squeeze(img, axis=2)
img = Image.fromarray(img)

img.save("modifier.png")