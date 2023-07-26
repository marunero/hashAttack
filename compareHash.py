import numpy as np

# # 3차원 이미지 크기 지정
# width = 256
# height = 256
# depth = 5  # 생성할 noise 개수

# # # normal distribution N(0,1) 생성
# # arr_list = [np.random.normal(loc=0, scale=1, size=(width, height))[:, :, np.newaxis] for i in range(depth)]
# # arr = np.concatenate(arr_list, axis=2)

# # # 확인을 위한 출력
# # print(arr.shape)

# a = np.random.normal(loc=0, scale=1, size=(width, height, 1))



# np.array([np.random.normal(loc = 0, scale = 1, size = (32, 32, 1)) for i in range(depth)])

# print(l.shape)

import os.path
import math
import cv2

import numpy as np
import tensorflow.compat.v1 as tf

from PIL import Image
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

libPath = './pyPhotoDNA/PhotoDNAx64.dll'

import pdqhash
import imagehash
import robusthash

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

# modifier = np.load("result/06_21_19_16_scaled_modifier.npy")

# while True:
#     ImageID = int(input("Image id = "))
    
#     # targetImageID = int(input("Target Image id = "))
#     targetImageID = 1


#     img1 = Image.open('InputImages/id0{:03d}.png'.format(ImageID)).convert("RGB")
#     img2 = Image.open('targetImages/target1.png'.format(targetImageID)).convert("RGB")

#     # img3 = np.array(img1)
#     # img3 = img3 + modifier * 255
#     # img3 = np.clip(img3, 0, 255).astype(np.uint8) 
#     # img3 = Image.fromarray(img3)

#     h1 = generatePhotoDNAHash(img1)
#     h2 = generatePhotoDNAHash(img2)

#     # new_h1 = generatePhotoDNAHash(img3)
#     print('photoDNA difference = ', PhotoDNA_Distance(h1, h2))
#     # print('photoDNA difference with modifier = ', PhotoDNA_Distance(new_h1, h2))

#     np_img1 = np.array(img1)
#     cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
#     h1, q1 = pdqhash.compute(cv2_img1)

#     np_img2 = np.array(img2)
#     cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
#     h2, q2 = pdqhash.compute(cv2_img2)

#     differ = ((h1 != h2) * 1).sum()

#     print('PDQ differ = ', differ)


#     h1 = imagehash.phash(img1, hash_size=16)
#     h2 = imagehash.phash(img2, hash_size=16)

#     print('phash differ = ', h1 - h2)

def pdq(img):
  np_img1 = np.array(img)
  cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
  h1, q1 = pdqhash.compute(cv2_img1)

  return h1

def pdq_differ(h1, h2):
  return ((h1 != h2) * 1).sum()



import imagehash

# img1 = Image.open(r"C:\Users\sungwoo\Downloads\test_searchengine\1_search.png")
# img3 = Image.open(r"C:\Users\sungwoo\Downloads\test_searchengine\1_top.jpg")
# img2 = Image.open(r"C:\Users\sungwoo\Downloads\test_searchengine\2_search.jpg")
# img4 = Image.open(r"C:\Users\sungwoo\Downloads\test_searchengine\2_top.jpg")

# h1 = imagehash.average_hash(img1)
# h2 = imagehash.average_hash(img2)
# h3 = imagehash.average_hash(img3)
# h4 = imagehash.average_hash(img4)
# print("original change ahash = ", h1 - h2)
# print("modified change ahash = ", h3 - h4)


# h1 = imagehash.phash(img1)
# h2 = imagehash.phash(img2)
# h3 = imagehash.phash(img3)
# h4 = imagehash.phash(img4)
# print("original change phash = ", h1 - h2)
# print("modified change phash = ", h3 - h4)


# h1 = imagehash.whash(img1)
# h2 = imagehash.whash(img2)
# h3 = imagehash.whash(img3)
# h4 = imagehash.whash(img4)
# print("original change whash = ", h1 - h2)
# print("modified change whash = ", h3 - h4)

# h1 = imagehash.dhash(img1)
# h2 = imagehash.dhash(img2)
# h3 = imagehash.dhash(img3)
# h4 = imagehash.dhash(img4)
# print("original change dhash = ", h1 - h2)
# print("modified change dhash = ", h3 - h4)

# # h3 = imagehash.phash(img1, hash_size=16)
# # h4 = imagehash.phash(img2, hash_size=16)
# # print(h3 - h4)

# h1 = imagehash.colorhash(img1, binbits=16)
# h2 = imagehash.colorhash(img2, binbits=16)
# h3 = imagehash.colorhash(img3, binbits=16)
# h4 = imagehash.colorhash(img4, binbits=16)
# print("original change colorhash = ", h1 - h2)
# print("modified change colorhash = ", h3 - h4)

# h1 = pdq(img1)
# h2 = pdq(img2)
# h3 = pdq(img3)
# h4 = pdq(img4)
# print("original change PDQ = ", pdq_differ(h1, h2))
# print("modified change PDQ = ", pdq_differ(h3, h4))

# img2 = np.array(img2)
# img2 = resize(img2,(img2.shape[0], img2.shape[1], 3), anti_aliasing=True)
# img2 -= 0.5
# modified_array2 = np.tanh(img2) / 2
# scaled_array2 = (modified_array2 + 0.5) * 255
# scaled_array2 = scaled_array2.astype(np.uint8)
# modified_img2 = Image.fromarray(scaled_array2)
# modified_img2.save('../test/modified_image2.png')



# img1 = np.array(img1)
# img1 = resize(img1,(img1.shape[0], img1.shape[1], 3), anti_aliasing=True)
# img1 -= 0.5
# modified_array1 = np.tanh(img1) / 2
# scaled_array1 = (modified_array1 + 0.5) * 255
# scaled_array1 = scaled_array1.astype(np.uint8)
# modified_img1 = Image.fromarray(scaled_array1)
# modified_img1.save('../test/modified_image1.png')


img1 = Image.open(r"C:\Users\sungwoo\Downloads\hashAttack\InputImages\id2025.png")
img2 = Image.open(r"C:\Users\sungwoo\Downloads\hashAttack\targetImages\target1.png")


# h1 = imagehash.average_hash(img1)
# h2 = imagehash.average_hash(img2)
# print("original change ahash = ", h1 - h2)
# print(h1, h2)

# h1 = imagehash.phash(img1)
# h2 = imagehash.phash(img2)
# print("original change phash = ", h1 - h2)
# print(h1, h2)



# h1 = imagehash.phash(img1, hash_size=16)
# h2 = imagehash.phash(img2, hash_size=16)
# print("original change phash256 = ", h1 - h2)
# print(h1, h2)


# h1 = imagehash.whash(img1)
# h2 = imagehash.whash(img2)
# print("original change whash = ", h1 - h2)

# h1 = imagehash.dhash(img1)
# h2 = imagehash.dhash(img2)
# print("original change dhash = ", h1 - h2)

# # h3 = imagehash.phash(img1, hash_size=16)
# # h4 = imagehash.phash(img2, hash_size=16)
# # print(h3 - h4)

# h1 = imagehash.colorhash(img1, binbits=16)
# h2 = imagehash.colorhash(img2, binbits=16)
# print("original change colorhash = ", h1 - h2)

# h1 = pdq(img1)
# h2 = pdq(img2)
# print("original change PDQ = ", pdq_differ(h1, h2))


h1 = generatePhotoDNAHash(img1)
h2 = generatePhotoDNAHash(img2)
print("change photoDNA = ", PhotoDNA_Distance(h1, h2))

h1 = robusthash.blockhash(img1.convert("RGB"))
print(len(h1))
h2 = robusthash.blockhash(img2.convert("RGB"))
print(sum(1 for i, j in zip(h1, h2) if i != j))

h1 = "b0cc247f334ce30f7c3c1a6e3521a31955d8a1e5"
h2 = "e166e687938e97414fb5848ba159bdfd45fdfc64"