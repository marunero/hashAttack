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


test_image_path = "C:/Users/sungwoo/Downloads/data_hashAttack/input/category2_test"
modifier_path = "C:/Users/sungwoo/Downloads/hashAttack/result/06_29_12_35_scaled_modifier.npy"
target_image_path = "C:/Users/sungwoo/Downloads/data_hashAttack/target/target2.png"

modifier = np.load(modifier_path)
target = Image.open(target_image_path)

metric = "PDQ"

if metric == "phash256":
  threshold = 90
elif metric == "photoDNA":
  threshold = 1800
# elif metric == "PDQ":
else:
  threshold = 90

if metric == "phash256":
  h2 = imagehash.phash(target, hash_size=16)
elif metric == "photoDNA":
  h2 = generatePhotoDNAHash(target)
else:
# elif metric == "PDQ":
  h2 = pdq(target)

img_list = os.listdir(test_image_path)

c = len(img_list)
success = 0
total_decrement = 0

for j, file_name in tqdm(enumerate(img_list)):
    # if i >= c:
    #   break
    img = Image.open(os.path.join(test_image_path, file_name)).convert("RGB")

    if metric == "phash256":
        h1 = imagehash.phash(img, hash_size=16)
        original_differ = h1 - h2
    elif metric == "photoDNA":
        h1 = generatePhotoDNAHash(img)
        original_differ = PhotoDNA_Distance(h1, h2)
    else:
    # elif metric == "PDQ":
        h1 = pdq(img)
        original_differ = pdq_differ(h1, h2)

    img = np.array(img)

    scaled_modifier = zoom(modifier, (img.shape[0] / modifier.shape[0], img.shape[1] / modifier.shape[1], 1))
    #   print(img.shape, scaled_modifier.shape)

    img = img + scaled_modifier * 255
    img = np.clip(img, 0, 255).astype(np.uint8) 
    img = Image.fromarray(img)

    if metric == "phash256":
        h1 = imagehash.phash(img, hash_size=16)
        transfer_differ = h1 - h2
    elif metric == "photoDNA":
        h1 = generatePhotoDNAHash(img)
        transfer_differ = PhotoDNA_Distance(h1, h2)
    else:
    # elif metric == "PDQ":
        h1 = pdq(img)
        transfer_differ = pdq_differ(h1, h2)
    
    total_decrement += transfer_differ - original_differ
    #   print(transfer_differ - original_differ)
    #   print("hash coolision ratio = ", success / (i + 1))
    if transfer_differ <= threshold:
        success += 1
        # print(file_name)
        # img.save(file_name)

print("total image number = ", c)
print("hash collision number = ", success)
print("hash collision ratio = ", success / c)
avg = total_decrement / c
print("average hash decrement = ", avg)
