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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import tensorflow.compat.v1 as tf

from PIL import Image
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

libPath = './pyPhotoDNA/PhotoDNAx64.dll'

import pdqhash

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


img1 = Image.open('InputImages/id00020.png')


# h1 = generatePhotoDNAHash(img1)
# ImageNet_path = "C:/Users/sungwoo/Downloads/imgdata"
# file_list = os.listdir(ImageNet_path)

# for i, file_name in enumerate(file_list):
#     if file_name == "id00020.png":
#         continue
#     img2 = Image.open(os.path.join(ImageNet_path, file_name))
    
#     h2 = generatePhotoDNAHash(img2)

#     if (PhotoDNA_Distance(h1, h2) == 0):
#         print(file_name)
#         print(PhotoDNA_Distance(h1, h2))



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



#-------------------------------------------------------------------------------------------------------------#
def divide(n, r, g, b):

    sum = int(r) + int(g) + int(b)
    if sum == 0:
      return round(n * 299/1000), round(n * 587/1000), n - round(n * 299/1000) - round(n * 587/1000)
    rp = round(r / sum * n)
    gp = round(g / sum * n)
    bp = n - rp - gp


    if rp + r > 255:
        overflow = rp + r - 255
        rp -= overflow

        gp += round(overflow / 2)
        bp += overflow - round(overflow / 2)
        if bp + b > 255:
          gp += bp + b - 255
    elif gp + g > 255:
        overflow = gp + g - 255
        gp -= overflow

        rp += round(overflow / 2)
        bp += overflow - round(overflow / 2)
        if bp + b > 255:
          rp += bp + b - 255
    elif bp + b > 255:
        overflow = bp + b - 255
        bp -= overflow

        rp += round(overflow / 2)
        gp += overflow - round(overflow / 2)
        if gp + g > 255:
          rp += rp + r - 255
    return rp, gp, bp
  

def addNoise2RGB(img, noise):
  # img = (375, 500, 3)
  # noise = (375, 500, 1)

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      r = img[i][j][0]
      g = img[i][j][1]
      b = img[i][j][2]

    #   rp, gp, bp = divide(round(noise[i][j][0]), r, g, b)
      rp, gp, bp = noise[i][j][0], noise[i][j][0], noise[i][j][0]

      img[i][j][0] += rp
      img[i][j][1] += gp
      img[i][j][2] += bp
      
  return img

def pdq(img):
  np_img1 = np.array(img)
  cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
  h1, q1 = pdqhash.compute(cv2_img1)

  return h1

def pdq_differ(h1, h2):
  return ((h1 != h2) * 1).sum()

modifier = np.load("result/06_14_21_08_scaled_modifier.npy")
target = Image.open("targetImages/id010.png")
h2 = pdq(target)

# input_i = Image.open("InputImages/id00017.png")
# h1 = pdq(input_i)
# print(pdq_differ(h1, h2))

# img1 = np.array(input_i)
# img1 = img1 + modifier * 255
# img1 = np.clip(img1, 0, 255).astype(np.uint8)

# cv2_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
# h1, q1 = pdqhash.compute(cv2_img1)

# print(pdq_differ(h1, h2))

# Image.fromarray(img1).save("test.png")

ImageNet_path = "C:/Users/sungwoo/Downloads/size_500_375"
img_list = os.listdir(ImageNet_path)

c = len(img_list)
success = 0
total_decrement = 0


for i, file_name in enumerate(img_list):
  # if i >= c:
  #   break
  img = Image.open(os.path.join(ImageNet_path, file_name)).convert("RGB")
  original_differ = pdq_differ(pdq(img), h2)

  img = np.array(img)
  img = img + modifier * 255
  img = np.clip(img, 0, 255).astype(np.uint8) 

  cv2_img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  h1, q1 = pdqhash.compute(cv2_img1)
  
  transfer_differ = pdq_differ(h1, h2)

  total_decrement += transfer_differ - original_differ
  # print(original_differ, transfer_differ)
  if transfer_differ <= 90:
    success += 1
    # print(file_name)
    # Image.fromarray(img).save(file_name)

print(success)
avg = total_decrement / c
print(avg)

    
    



# modifier = np.load("result/06_13_18_45_scaled_modifier.npy")
# modified_img = Image.open("result/photoDNA/06_02_17_41_True_inputID20_targetID20_photoDNAdiffer_1800.0_l2distance_13373.6435546875k.png")
# input_img = Image.open("InputImages/id00020.png")
# target_img = Image.open("targetImages/id020.png")

# input_np = np.array(input_img)

# new = addNoise2RGB(input_np, modifier * 255)
# modifier = np.concatenate((modifier, modifier, modifier), axis=2)
# new = input_img + (modifier * 255) 

# for i in range(new.shape[0]):
#   for j in range(new.shape[1]):
#     r = new[i][j][0]
#     g = new[i][j][1]
#     b = new[i][j][2]

#     if r > 255:
#       over = r - 255
#       r = 255
#       g += over / 2
#       b += over / 2
#       if g > 255:
#         over = g - 255
#         g = 255
#         b += over
#       elif b > 255:
#         over = b - 255
#         b = 255
#         g += over

#     elif g > 255:
#       over = g - 255
#       g = 255
#       r += over / 2
#       b += over / 2
#       if r > 255:
#         over = r - 255
#         r = 255
#         b += over
#       elif b > 255:
#         over = b - 255
#         b = 255
#         r += over
    
#     elif b > 255:
#       over = b - 255
#       b = 255
#       r += over / 2
#       g += over / 2
#       if r > 255:
#         over = r - 255
#         r = 255
#         g += over
#       elif g > 255:
#         over = g - 255
#         g = 255
#         r += over

#     new[i][j][0] = r
#     new[i][j][1] = g
#     new[i][j][2] = b
    
#     print(i, j)
    

  
# new = np.clip(new, 0, 255)
# new = new.astype(np.uint8)

# new_img = Image.fromarray(new)
# new_img.save("test.png")


# h3 = generatePhotoDNAHash(new_img)


# h1 = generatePhotoDNAHash(target_img)
# h2 = generatePhotoDNAHash(modified_img)

# print("photoDNA difference grayscale = ", PhotoDNA_Distance(h1, h2))
# print("photoDNA difference RGB = ", PhotoDNA_Distance(h1, h3))



# np_img1 = np.array(new_img)
# cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
# h1, q1 = pdqhash.compute(cv2_img1)

# np_img2 = np.array(target_img)
# cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
# h2, q2 = pdqhash.compute(cv2_img2)

# np_img3 = np.array(modified_img)
# cv2_img3 = cv2.cvtColor(np_img3, cv2.COLOR_RGB2BGR)
# h3, q3 = pdqhash.compute(cv2_img3)

# differ12 = ((h1 != h2) * 1).sum()
# differ23 = ((h2 != h3) * 1).sum()

# print("PDQ difference grayscale = ", differ23)
# print("PDQ difference RGB = ", differ12)

# print(PhotoDNA_Distance(h1, h3))
# #-------------------------------------------------------------------------------------------------------------#


# noise = np.random.normal(loc=0, scale = 16, size=(375, 500))



# resized = resize(noise, (375, 500), anti_aliasing = True)

# print(resized.max())
# # img2 = resize(img2,(img2.shape[0], img2.shape[1], 3), anti_aliasing=True)

# noise = noise.astype('uint8')

# img = Image.fromarray(noise)
# img.save('noise.png')



# resized = (resized).astype('uint8')

# img2 = Image.fromarray(resized)
# img2.save('noise_resized.png')




# data = np.load("result/05_30_16_50.npy")

# data = np.squeeze(data, axis=2)

# nrows, ncols = data.shape

# # 3D 그래프 생성
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 배열의 인덱스를 사용하여 3D 그래프에 데이터 포인트 추가
# for row in range(nrows):
#     for col in range(ncols):
#         ax.scatter(row, col, data[row, col])

# # 축 레이블 설정
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 그래프 표시
# plt.show()


# img1 = Image.open('result/06_07_02_05_True_5019_inputID17_targetID17_lr0.01_pc16_pdq_photoDNAdiffer3600.0_l2distance_12120.275390625.png')
# img2 = Image.open('targetImages/id017.png')


# h1 = generatePhotoDNAHash(img1)
# h2 = generatePhotoDNAHash(img2)

# print(PhotoDNA_Distance(h1, h2))

# np_img1 = np.array(img1)
# cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
# h1, q1 = pdqhash.compute(cv2_img1)


# np_img2 = np.array(img2)
# cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
# h2, q2 = pdqhash.compute(cv2_img2)

# differ = ((h1 != h2) * 1).sum()

# print("PDQ difference = ", differ)