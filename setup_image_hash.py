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
    arr = np.clip(arr, -0.5, 0.5)
    fig = np.around((arr + 0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img

def gen_image_cv2(arr):
    arr = np.clip(arr, -0.5, 0.5)
    fig = np.around((arr + 0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()

    cv2_image = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)

    return cv2_image

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

def loss_phash64(inputs, target, targeted):
    a = []
    h1 = imagehash.phash(gen_image(target))


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.phash(gen_image(inputs[i]))

        a.append(h1 - h2)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_phash256(inputs, target, targeted):
    a = []
    h1 = imagehash.phash(gen_image(target), hash_size=16)


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.phash(gen_image(inputs[i]), hash_size=16)

        a.append(h1 - h2)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_ahash64(inputs, target, targeted):
    a = []
    h1 = imagehash.average_hash(gen_image(target))


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.average_hash(gen_image(inputs[i]))

        a.append(h1 - h2)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_ahash256(inputs, target, targeted):
    a = []
    h1 = imagehash.average_hash(gen_image(target), hash_size=16)


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.average_hash(gen_image(inputs[i]), hash_size=16)

        a.append(h1 - h2)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_photoDNA(inputs, target, targeted):
    a = []

    h1 = generatePhotoDNAHash(gen_image(target))

    for i in range(inputs.shape[0]):
        h2 = generatePhotoDNAHash(gen_image(inputs[i]))

        a.append(PhotoDNA_Distance(h1, h2))

    a = np.asarray(a, dtype=np.float32)
    return a
    
def loss_colorhash(inputs, target, targeted):
    a = []
    h1 = imagehash.colorhash(gen_image(target), binbits=16)


    for i in range(inputs.shape[0]):
        # black-box
        h2 = imagehash.colorhash(gen_image(inputs[i]), binbits=16)

        a.append(h1 - h2)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_imagehash_comb(inputs, target, targeted):
    a = []
    h1_color = imagehash.colorhash(gen_image(target), binbits=16)
    h1_ahash = imagehash.average_hash(gen_image(target), hash_size=16)
    h1_phash = imagehash.phash(gen_image(target), hash_size=16)
    h1_whash = imagehash.whash(gen_image(target), hash_size=16)
    h1_dhash = imagehash.dhash(gen_image(target), hash_size=16)
    

    for i in range(inputs.shape[0]):
        h2_color = imagehash.colorhash(gen_image(inputs[i]), binbits=16)
        h2_ahash = imagehash.average_hash(gen_image(inputs[i]), hash_size=16)
        h2_phash = imagehash.phash(gen_image(inputs[i]), hash_size=16)
        h2_whash = imagehash.whash(gen_image(inputs[i]), hash_size=16)
        h2_dhash = imagehash.dhash(gen_image(inputs[i]), hash_size=16)

        dif = h1_color - h2_color
        dif += h1_ahash - h2_ahash
        dif += h1_phash - h2_phash
        dif += h1_whash - h2_whash
        dif += h1_dhash - h2_dhash

        a.append(dif)
        
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_PDQ(inputs, target, targeted):
    a = []

    img1 = gen_image(target)
    np_img1 = np.array(img1, dtype=np.uint8)
    cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    h1, q1 = pdqhash.compute(cv2_img1)

    for i in range(inputs.shape[0]):
        img2 = gen_image(inputs[i])
        np_img2 = np.array(img2, dtype=np.uint8)
        cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
        h2, q2 = pdqhash.compute(cv2_img2)

        differ = ((h1 != h2) * 1).sum()
        
        # a.append(max(0, np.tanh(differ / 256))) 
        a.append(differ)
        
    
    a = np.asarray(a, dtype=np.float32)
    return a

def loss_PDQ_photoDNA(inputs, target, targeted):
    a = []

    h1_photoDNA = generatePhotoDNAHash(gen_image(target))

    img1 = gen_image(target)    
    np_img1 = np.array(img1, dtype=np.uint8)
    cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    h1_PDQ, q1 = pdqhash.compute(cv2_img1)

    for i in range(inputs.shape[0]):
        h2_photoDNA = generatePhotoDNAHash(gen_image(inputs[i]))

        img2 = gen_image(inputs[i])
        np_img2 = np.array(img2, dtype=np.uint8)
        cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
        h2_PDQ, q2 = pdqhash.compute(cv2_img2)

        differ_PDQ = ((h1_PDQ != h2_PDQ) * 1).sum()
        differ_photoDNA = PhotoDNA_Distance(h1_photoDNA, h2_photoDNA)
        
        # a.append(max(0, np.tanh(differ / 256))) 
        a.append(max(differ_PDQ, 90) * 20 + max(differ_photoDNA, 1800))
        

    a = np.asarray(a, dtype=np.float32)
    return a

def calculate_weighted_count(arr):
    weight_intervals = [(0, 50, 16), (50, 100, 8), (100, 150, 4), (150, 200, 2), (200, 250, 1), (250, 500, 0.5)]
    weighted_count = 0

    for value in arr:
        for start, end, weight in weight_intervals:
            if start <= value <= end:
                weighted_count += weight * (250 - value)
                break  # Break out of the inner loop once the range is found

    return weighted_count

def loss_sift(inputs, target, targeted):
    # a = []

    # value_i = 0
    # value_t = 0
    # const = 0.00001

    # detector = cv2.ORB_create(nfeatures=153)

    # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # kp1, desc1 = detector.detectAndCompute(gen_image_cv2(target), None)
    # kp2, desc2 = detector.detectAndCompute(gen_image_cv2(inputs[0]), None)

    # for i in range(1, inputs.shape[0]):
        
    #     kp3, desc3 = detector.detectAndCompute(gen_image_cv2(inputs[i]), None)

    #     matches = matcher.match(desc1, desc3)        
    #     matches_distance = [match.distance for match in matches]        
    #     value_t = len(matches) - sum(matches_distance) * const

    #     matches = matcher.match(desc2, desc3)
    #     matches_distance = [match.distance for match in matches]
    #     value_i = len(matches) - sum(matches_distance) * const

    #     # value_i = calculate_weighted_count(matches_distance_input) / len(matches_distance_input)
    #     # value_t = calculate_weighted_count(matches_distance_target) / len(matches_distance_target)

    #     a.append(value_i - 3 * value_t)

    # a = np.asarray(a, dtype=np.float32)
    # return a

    a = []

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=153)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    kp1, desc1 = sift.detectAndCompute(gen_image_cv2(target), None)
    kp2, desc2 = sift.detectAndCompute(gen_image_cv2(inputs[0]), None)

    for i in range(1, inputs.shape[0]):        
        kp3, desc3 = sift.detectAndCompute(gen_image_cv2(inputs[i]), None)

        matches = flann.knnMatch(desc1, desc3, k = 2)

        count_t = 0
        count_i = 0
        # ratio test as per Lowe's paper
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.35 * n.distance:
                count_t += 1

        
        matches = flann.knnMatch(desc2, desc3, k = 2)
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.35 * n.distance:
                count_i += 1

        a.append(count_i - count_t)

    a = np.asarray(a, dtype=np.float32)
    return a




def read_inputImage(ff):
  f = inputImages_path + ff

  img = Image.open(f)
  img_gray = img.convert("L")
  img = np.array(img, dtype=np.uint8)
  gray_img = np.array(img_gray, dtype=np.uint8)

  img = resize(img,(img.shape[0], img.shape[1], 3), anti_aliasing=True)
  gray_img = resize(gray_img,(gray_img.shape[0],gray_img.shape[1], 1), anti_aliasing=True)
  img -= 0.5
  gray_img -= 0.5

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
        img = np.array(img, dtype=np.uint8)
        img = resize(img, (img.shape[0], img.shape[1], 3), anti_aliasing = True)
        img -= 0.5

        target_data.append(img)

    self.input_images_rgb = np.array(input_images_rgb, dtype=np.float32)
    self.input_images_gray = np.array(input_images_gray, dtype=np.float32)
    self.target_images_rgb = np.array(target_data, dtype=np.float32)

class ImageNet_Hash:
    def __init__(self, targeted = True):
      self.targeted = targeted
    
    def get_loss(self, inputs, target, method):
        if method == "phash64":
            return tf.py_function(loss_phash64, [inputs, target, self.targeted], tf.float32)
        elif method == "phash256":
            return tf.py_function(loss_phash256, [inputs, target, self.targeted], tf.float32)
        elif method == "colorhash":
            return tf.py_function(loss_colorhash, [inputs, target, self.targeted], tf.float32)
        elif method == "ahash64":
            return tf.py_function(loss_ahash64, [inputs, target, self.targeted], tf.float32)
        elif method == "ahash256":
            return tf.py_function(loss_ahash256, [inputs, target, self.targeted], tf.float32)
        elif method == "pdqhash":
            return tf.py_function(loss_PDQ, [inputs, target, self.targeted], tf.float32)
        elif method == "photoDNA":
            return tf.py_function(loss_photoDNA, [inputs, target, self.targeted], tf.float32)
        elif method == "pdq_photoDNA":
            return tf.py_function(loss_PDQ_photoDNA, [inputs, target, self.targeted], tf.float32)
        elif method == "imagehash_comb":
            return tf.py_function(loss_imagehash_comb, [inputs, target, self.targeted], tf.float32)
        elif method == "SIFT":
            return tf.py_function(loss_sift, [inputs, target, self.targeted], tf.float32)
        # else
        else:
            return tf.py_function(loss_photoDNA, [inputs, target, self.targeted], tf.float32)


