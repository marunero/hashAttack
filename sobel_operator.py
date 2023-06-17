import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt

from PIL import Image

import imagehash
import pdqhash
import cv2
# photoDNA
import glob
import base64
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p
libPath = './pyPhotoDNA/PhotoDNAx64.dll'


def generatePhotoDNAHash(imageFile):
    if imageFile.mode != 'RGB':
        imageFile = imageFile.convert(mode = 'RGB')
    libPhotoDNA = cdll.LoadLibrary(libPath)
    ComputeRobustHash = libPhotoDNA.ComputeRobustHash
    ComputeRobustHash.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
    ComputeRobustHash.restype = c_ubyte
    hashByteArray = (c_ubyte * 144)()
    ComputeRobustHash(c_char_p(imageFile.tobytes()), imageFile.width, imageFile.height, 0, hashByteArray, 0)
    hashPtr = cast(hashByteArray, POINTER(c_ubyte))
    hashList = [str(hashPtr[i]) for i in range(144)]
    hashString = ','.join([i for i in hashList])
    hashList = hashString.split(',')
    for i, hashPart in enumerate(hashList):
        hashList[i] = int(hashPart).to_bytes((len(hashPart) + 7) // 8, 'big')
    hashBytes = b''.join(hashList)
    return hashBytes

def PhotoDNA_Distance(h1, h2):
	distance = 0
	if len(h1) != 144:
		panic("h1 wrong length")
	
	if len(h2) != 144:
		panic("h2 wrong length")
	

	for i in range(len(h1)):
		distance += abs(h1[i] - h2[i])
	return distance


# Here we read the image and bring it as an array
original_image = imread('InputImages/id00011.png')

# Next we apply the Sobel filter in the x and y directions to then calculate the output image
dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step


filer = np.array(sobel_filtered_image)
img = np.array(original_image)

print(filer.min(), filer.max())

while True:
    const = int(input("enter const "))
    filer = np.array(sobel_filtered_image)
    img = np.array(original_image)  
    new_img = filer * const + img * 255

    print(np.sqrt(np.sum((filer * 255) ** 2)))

    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)
    new_img = Image.fromarray(new_img)


    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    print("PhotoDNA", PhotoDNA_Distance(generatePhotoDNAHash(img), generatePhotoDNAHash(new_img)))


    np_img1 = np.array(img)
    cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    h1, q1 = pdqhash.compute(cv2_img1)

    np_img2 = np.array(new_img)
    cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
    h2, q2 = pdqhash.compute(cv2_img2)

    print("PDQ", ((h1 != h2) * 1).sum())

    # Display and compare input and output images
    fig = plt.figure(1)
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.imshow(original_image)
    ax2.imshow(new_img)
    plt.show()