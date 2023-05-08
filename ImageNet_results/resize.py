from __future__ import absolute_import, division, print_function
from torchvision.transforms import ToTensor, ToPILImage

import sys

import os
from PIL import Image
from torchvision import transforms
import imagehash
import torch 
import robusthash
import cv2
import tensorflow.compat.v1 as tf
import pandas as pd

from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
from PIL import Image, ImageFilter
try:
	ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
	# deprecated in pillow 10
	# https://pillow.readthedocs.io/en/stable/deprecations.html
	ANTIALIAS = Image.ANTIALIAS


import shutil
from matplotlib import pyplot as plt
import blockhash

import pdqhash


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



def grad_test():

	img1 = Image.open("ImageNet_sorted3/id00000.png")

	np_img1 = np.array(img1)
	cv2_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
	h1, q1 = pdqhash.compute(cv2_img1)

	img2 = Image.open("ImageNet_sorted3/id00001.png")
	np_img2 = np.array(img2)
	cv2_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
	h2, q2 = pdqhash.compute(cv2_img2)

	print(h1)
	print(len(h1))
	h1_float, _ = pdqhash.compute_float(cv2_img1)
	print(h1_float.max())
	print(h1_float.min())

grad_test()