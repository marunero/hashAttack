from tqdm import tqdm

import os.path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import zoom
import numpy as np
import tensorflow.compat.v1 as tf

from PIL import Image
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

libPath = './pyPhotoDNA/PhotoDNAx64.dll'

import pdqhash
import imagehash

# photoDNA
import glob
import base64
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

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

def pdq(img):
  np_img1 = np.array(img)
  cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
  h1, q1 = pdqhash.compute(cv2_img1)

  return h1

def pdq_differ(h1, h2):
  return ((h1 != h2) * 1).sum()


image_path = r"C:\Users\sungwoo\Downloads\\test_searchengine\test.png"
modifier_path = "C:/Users/sungwoo/Downloads/hashAttack/result/old/06_21_21_30_scaled_modifier_loss_max_90.npy"

modifier = np.load(modifier_path)
img = Image.open(image_path).convert("RGB")

img = np.array(img)
scaled_modifier = zoom(modifier, (img.shape[0] / modifier.shape[0], img.shape[1] / modifier.shape[1], 1))

img = img + scaled_modifier * 255 * 2
img = np.clip(img, 0, 255).astype(np.uint8) 

img = Image.fromarray(img)

img = img.rotate(10)

img.save("modified_img.png")

