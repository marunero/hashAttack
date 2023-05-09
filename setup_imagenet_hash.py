from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import random
import tarfile
import scipy.misc

import numpy as np
from six.moves import urllib
import tensorflow as tf
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import imagehash
import robusthash
import math
import cv2

import pdqhash

try:
	ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
	# deprecated in pillow 10
	# https://pillow.readthedocs.io/en/stable/deprecations.html
	ANTIALIAS = Image.ANTIALIAS


# PhotoDNA
import glob
import time
import base64
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

libPath = r'C:\Users\sungwoo\Downloads\hashAttack\pyPhotoDNA\PhotoDNAx64.dll'

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

global_threshold = 0

def phash(image, hash_size=8, highfreq_factor=4):
	# type: (Image.Image, int, int) -> ImageHash
	"""
	Perceptual Hash computation.
	Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
	@image must be a PIL instance.
	"""
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
	pixels = np.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = np.median(dctlowfreq)
	diff = dctlowfreq > med
	return dctlowfreq, med, diff


def gen_image(arr):
    # two_d = (np.reshape(arr, (28, 28)) * 255 ).astype(np.uint8)
    #
    # img = Image.fromarray(two_d)
    fig = np.around((arr+0.5) * 255.0)
    # fig = (arr + 0.5) * 255
    # fig = fig.astype(np.uint8).squeeze()
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img

def pfunction(arr, arr1):
    a = []
    differences = []

    for i in range(arr.shape[0]):

        a.append(max(0, 1-  ((imagehash.phash(gen_image(arr[i])) -  imagehash.phash(gen_image(arr1)) ) / 8 )  ))
    #     differences.append(imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1)))
    # print('how is it possible ', differences)
    a = np.asarray(a)
    a = a.astype('float32')

    return a


def pfunction_log(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append(max(0, -math.log((imagehash.phash(gen_image(arr[i])) -  imagehash.phash(gen_image(arr1)) ) + 1e-30) ))

    a = np.asarray(a)
    a = a.astype(float)
    return a
def pfunction_square(arr, arr1):
    a = []
    for i in range(arr.shape[0]):
        a.append((max(0, 1 - ((imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1))) / 8)))**2)

    a = np.asarray(a)
    a = a.astype('float32')
    return a

box = "gray"

def pfunction_tanh(arr, arr1, targeted):
    a = []
        
    f1, m1, d1 = phash(gen_image(arr1))
    f2, m2, d2 = phash(gen_image(arr1))
    
    for i in range(arr.shape[0]):
        # black-box
        # if targeted == False:
        #     a.append(max(0, np.tanh(1 - ((imagehash.phash(gen_image(arr[i]), global_bits, global_factor) - imagehash.phash(gen_image(arr1), global_bits, global_factor)) / global_threshold))))
        # else:
        #     a.append(max(0, np.tanh(((imagehash.phash(gen_image(arr[i]), global_bits, global_factor) - imagehash.phash(gen_image(arr1), global_bits, global_factor)) / global_threshold)))) 

        # gray-box
        if targeted == False:
            f1, m1, d1 = phash(gen_image(arr[i]))

            if ((d1 != d2) * 1).sum() >= global_threshold:
                a.append(0)
            else:
                a.append((np.power((f1 - m1) * ((d1 == d2) * 1), 2).sum()) ** 0.5)

                # a.append((((f1 - m1) * ((d1 == d2) * 1)).sum()))
        
        else:
            f1, m1, d1 = phash(gen_image(arr[i]))
            a.append((np.power((f1 - m1) * ((d1 != d2) * 1), 2).sum()) ** 0.5)

            # a.append((((f1 - m1) * ((d1 != d2) * 1)).sum()) ** 2)
            # a.append((abs(f1 - m1) * (d1 != d2) * 1).sum())

    a = np.asarray(a)
    a = a.astype('float32')
    return a


def pfunction_tanh_pdq(arr, arr1, targeted):
    a = []
    img1 = gen_image(arr1)
    np_img1 = np.array(img1)
    cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    h1, q1 = pdqhash.compute(cv2_img1)

    if box == "black":
        for i in range(arr.shape[0]):
            img2 = gen_image(arr[i])
            np_img2 = np.array(img2)
            cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
            h2, q2 = pdqhash.compute(cv2_img2)

            differ = ((h1 != h2) * 1).sum()

            if targeted == False:
                a.append(max(0, np.tanh(1 - differ / 256)))
            else:
                a.append(max(0, np.tanh(differ / 256))) 
    
    elif box == "gray":
        h1_float, _ = pdqhash.compute_float(cv2_img1)
        for i in range(arr.shape[0]):
            img2 = gen_image(arr[i])
            np_img2 = np.array(img2)
            cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
            h2, q2 = pdqhash.compute(cv2_img2)

            if targeted == False:
                a.append((np.power((h1_float) * ((h1 == h2) * 1), 2).sum()) ** 0.5)
                
            else:
                a.append((np.power((h1_float) * ((h1 != h2) * 1), 2).sum()) ** 0.5)

                # a.append((((f1 - m1) * ((d1 != d2) * 1)).sum()) ** 2)
                # a.append((abs(f1 - m1) * (d1 != d2) * 1).sum())
                
    a = np.asarray(a)
    a = a.astype('float32')
    return a

def pfunction_tanh_photoDNA(arr, arr1, targeted):
    a = []

    h1 = generatePhotoDNAHash(gen_image(arr1))

    for i in range(arr.shape[0]):
        h2 = generatePhotoDNAHash(gen_image(arr[i]))

        a.append(PhotoDNA_Distance(h1, h2))

    a = np.asarray(a)
    a = a.astype('float32')
    return a

def pfunction_tanh2(arr, arr1, targeted):
    a = []
    timage = gen_image(arr1)
    if timage.mode == '1' or timage.mode == 'L' or timage.mode == 'P':
        timage = timage.convert('RGB')
    for i in range(arr.shape[0]):
        newimage = gen_image(arr[i])
        if newimage.mode == '1' or newimage.mode == 'L' or newimage.mode == 'P':
            newimage = newimage.convert('RGB')
        #a.append(max(0, np.tanh(1 - ((imagehash.average_hash(gen_image(arr[i])) - imagehash.average_hash(gen_image(arr1))) /global_threshold))))
        if targeted == False:
            a.append(max(0, np.tanh( 1 - sum(1 for i, j in zip(robusthash.blockhash(newimage), robusthash.blockhash(timage)) if i != j)/ global_threshold )))
        else:
            a.append(max(0, np.tanh(sum(1 for i, j in zip(robusthash.blockhash(newimage), robusthash.blockhash(timage)) if i != j)/ global_threshold )))
    a = np.asarray(a)
    a = a.astype('float32')
    return a

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pfunction_sigmoid(arr, arr1):
    a = []
    differences = []
    for i in range(arr.shape[0]):
        a.append(max(0, sigmoid(1 - ((imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1))) / 8)) -0.5))
        differences.append(imagehash.phash(gen_image(arr[i])) - imagehash.phash(gen_image(arr1)))
    print('how is it possible ', differences)
    a = np.asarray(a)
    a = a.astype('float32')
    return a

def readimg(ff):
  f = "./ImageNet_sorted3/"+ff
  #f = "./ImageNet_sorted/"+ff
  img = Image.open(f)
  img_gray = img.convert("L")
  img = np.array(img)
  gray_img = np.array(img_gray)
#   img = resize(img,(299,299))-.5
  # skip small images (image should be at least 299x299)
  
  img = resize(img,(img.shape[0], img.shape[1], 3), anti_aliasing=True)
  gray_img = resize(gray_img,(gray_img.shape[0],gray_img.shape[1], 1), anti_aliasing=True)

#   rgb_weights = [0.2989, 0.5870, 0.1140]
#   gray_img = np.dot(img[...,:3], rgb_weights).reshape((299,299,1)) - 0.5
  gray_img = gray_img - 0.5
  img = img - 0.5
#   if img.shape != (288, 288, 3):
#     return None
  return [img, gray_img]

def read_target(ff):
    f = "./targetImage/" + ff
    img = Image.open(f)
    img = np.array(img)
    img = resize(img, (img.shape[0], img.shape[1], 3), anti_aliasing = True)
    img -= 0.5
    return img

class ImageNet:
  def __init__(self):
    from multiprocessing import Pool
    pool = Pool(8)
    file_list = sorted(os.listdir("./ImageNet_sorted3"))
    target_list = sorted(os.listdir("./targetImage"))
    #file_list = sorted(os.listdir("./ImageNet_sorted/"))
    # random.seed(2020)
    # random.shuffle(file_list)
    r = pool.map(readimg, file_list)
    # print(file_list[:200])
    r = [x for x in r if x != None]
    # test_data, test_labels = zip(*r)
    

    test_data, test_data_gray = zip(*r)
    
    target_data = []
    for i in range(len(target_list)):
        f = "./targetImage/" + target_list[i]
        img = Image.open(f)
        img = np.array(img)
        img = resize(img, (img.shape[0], img.shape[1], 3), anti_aliasing = True)
        img -= 0.5
        target_data.append(img)

    # print('how do you get labels of', test_labels)
    self.test_data = np.array(test_data)
    # self.test_labels = np.zeros((len(test_labels), 1001))
    self.test_data_gray = np.array(test_data_gray)
    self.target_data = np.array(target_data)
    # self.test_labels[np.arange(len(test_labels)), test_labels] = 1
    # print('2 imagenet image shape ', self.test_data.shape)
    # print('2 imagenet image label shape ', self.test_labels.shape)


class ImageNet_HashModel:
    def __init__(self, hash, bits, factor):
        # self.num_channels = 1
        # self.image_width = 288
        # self.image_height = 288
        # self.num_labels = 10
        global global_threshold
        global_threshold = hash
        global global_bits
        global_bits = bits
        global global_factor
        global_factor = factor
    
    def predict1(self, data, data1, method, targeted):

        if method == 'linear':
            return tf.py_function(pfunction, [data, data1], tf.float32)
        elif method == 'square':
            return tf.py_function(pfunction_square, [data, data1], tf.float32)
        elif method =='tanh':
            return tf.py_function(pfunction_tanh, [data, data1, targeted], tf.float32)
        elif method == 'sigmoid':
            return tf.py_function(pfunction_sigmoid, [data, data1], tf.float32)

    def predict2(self, data, data1, targeted):
        return tf.py_function(pfunction_tanh2, [data, data1, targeted], tf.float32)
        
    def predict_pdq(self, data, data1, targeted):
        return tf.py_function(pfunction_tanh_pdq, [data, data1, targeted], tf.float32)

    def predict_photoDNA(self, data, data1, targeted):
        return tf.py_function(pfunction_tanh_photoDNA, [data, data1, targeted], tf.float32)
