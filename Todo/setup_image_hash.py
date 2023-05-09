from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import math
import cv2

import numpy as np

from PIL import Image

import imagehash
import pdqhash
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

inputImages_path = "./InputImages"
targetImages_path = "./targetImages"

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
    self.target_images_rgb = np.array(target_images_rgb)

class ImageNet_Hash:
    def __init__(self):
        
    
    