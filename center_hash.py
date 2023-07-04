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

# def hex_to_binary(hex_value):
#     binary_value = bin(int(hex_value, 16))[2:]  # Convert hex to binary and remove '0b' prefix
#     return binary_value.zfill(len(hex_value) * 4)  # Ensure the binary representation has the same length

# def hamming_distance(hex1, hex2):
#     bin1 = hex_to_binary(hex1)
#     bin2 = hex_to_binary(hex2)

#     distance = sum(bit1 != bit2 for bit1, bit2 in zip(bin1, bin2))
#     return distance


img_dir = r"D:\ILSVRC2012_img_train\affenpinscher "

img_list = os.listdir(img_dir)

total = 0
for i, file_name in tqdm(enumerate(img_list)):
    if i > 16:
        break
    img = Image.open(os.path.join(img_dir, file_name))
    h = pdq(img)

    bit_string = ''.join(str(bit) for bit in h)
    integer_value = int(bit_string, 2)
    
    total += integer_value / len(img_list)
    # print(total)

center = int(total)

center = bin(center)[2:]

print(center)

target_img_dir = r"D:\ILSVRC2012_img_train\affenpinscher "

target_img_list = os.listdir(target_img_dir)


total = 0
for i, file_name in tqdm(enumerate(target_img_list)):
    img = Image.open(os.path.join(target_img_dir, file_name))
    h = pdq(img)

    bit_string = ''.join(str(bit) for bit in h)
    
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(bit_string, center))
    total += distance ** 2

var = total ** 0.5
print("variance = ", var)
