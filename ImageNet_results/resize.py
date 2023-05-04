from __future__ import absolute_import, division, print_function
from torchvision.transforms import ToTensor, ToPILImage

import sys

import os
from PIL import Image
from torchvision import transforms
import imagehash
import torch 
import robusthash
import cv2 as cv
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

def resizeImage():
	path = "C:/Users/sungwoo/Downloads/imgdata"
	file_list = os.listdir(path)

	size_counts = {}

	index = 0

	for i, file_name in enumerate(file_list):
		img = Image.open(os.path.join(path, file_name))

		# big_size = (600, 600)
		# small_size = (200, 200)

		# big_image = img.resize(big_size)
		# small_image = img.resize(small_size)

		# big_image.save('resize_save/' + 'big_' + file_name)
		# small_image.save('resize_save/' + 'small_' + file_name

		# if img.size in size_counts:
		# 	size_counts[img.size] += 1
		# else:
		# 	size_counts[img.size] = 1
		if img.size == (500, 375):
			img.save('C:/Users/sungwoo/Downloads/size_500_375/' + "id{:05d}.png".format(index))
			index += 1



	# save_path = 'resize_save'

	# save_file_list = os.listdir(save_path)

	# for i, file_name in enumerate(save_file_list):
	# 	img = Image.open(os.path.join(path, file_name))



	

resizeImage()