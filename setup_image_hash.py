from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import math
import cv2

import numpy as np
import tensorflow as tf

from PIL import Image
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

import imagehash
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

inputImages_path = "./InputImages/"
targetImages_path = "./targetImages/"

libPath = './pyPhotoDNA/PhotoDNAx64.dll'

def gen_image(arr):
    fig = np.around((arr+0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img

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

def loss_photoDNA(inputs, target, targeted):
    a = []

    h1 = generatePhotoDNAHash(gen_image(target))

    for i in range(inputs.shape[0]):
        h2 = generatePhotoDNAHash(gen_image(inputs[i]))

        a.append(PhotoDNA_Distance(h1, h2))

    a = np.asarray(a)
    a = a.astype('float32')
    return a
    
def loss_phash(inputs, target, targeted):
    a = []
    h1 = imagehash.phash(gen_image(target))


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.phash(gen_image(inputs[i]))

        a.append(h1 - h2)
        
    a = np.asarray(a)
    a = a.astype('float32')
    return a


def loss_PDQ(inputs, target, targeted):
    a = []

    img1 = gen_image(target)
    np_img1 = np.array(img1)
    cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    h1, q1 = pdqhash.compute(cv2_img1)

    for i in range(inputs.shape[0]):
        img2 = gen_image(inputs[i])
        np_img2 = np.array(img2)
        cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
        h2, q2 = pdqhash.compute(cv2_img2)

        differ = ((h1 != h2) * 1).sum()
        
        # a.append(max(0, np.tanh(differ / 256))) 
        a.append(differ)
        
    
    a = np.asarray(a)
    a = a.astype('float32')
    return a


def read_inputImage(ff):
  f = inputImages_path + ff

  img = Image.open(f)
  img_gray = img.convert("L")
  img = np.array(img)
  gray_img = np.array(img_gray)

  img = resize(img,(img.shape[0], img.shape[1], 3), anti_aliasing=True)
  gray_img = resize(gray_img,(gray_img.shape[0],gray_img.shape[1], 1), anti_aliasing=True)

  gray_img = gray_img - 0.5
  img = img - 0.5

  return [img, gray_img]

class ImageNet:
  def __init__(self):
    from multiprocessing import Pool
    pool = Pool(8)

    file_list = sorted(os.listdir(inputImages_path))
    target_list = sorted(os.listdir(targetImages_path))
    r = pool.map(read_inputImage, file_list)
    r = [x for x in r if x != None]

    input_images_rgb, input_images_gray = zip(*r)

    target_data = []
    for i in range(len(target_list)):
        f = targetImages_path + target_list[i]
        img = Image.open(f)
        img = np.array(img)
        img = resize(img, (img.shape[0], img.shape[1], 3), anti_aliasing = True)
        img -= 0.5
        target_data.append(img)

    self.input_images_rgb = np.array(input_images_rgb)
    self.input_images_gray = np.array(input_images_gray)
    self.target_images_rgb = np.array(target_data)

class ImageNet_Hash:
    def __init__(self, targeted = True):
      self.targeted = targeted
    
    def get_loss_phash(self, inputs, target):
      return tf.py_function(loss_phash, [inputs, target, self.targeted], tf.float32)
    def get_loss_pdq(self, inputs, target):
      return tf.py_function(loss_PDQ, [inputs, target, self.targeted], tf.float32)
    def get_loss_photoDNA(self, inputs, target):
      return tf.py_function(loss_photoDNA, [inputs, target, self.targeted], tf.float32)

        
      
    
    