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



def grad_test():
	multi_imgs_num = 3
	batch_size = 4
	mc_sample = 4
	delta = 0.19999

	for i in range(batch_size):
		print("-----")
		print("batch = " + str(i))
		print("-----")
		
		for j in range(multi_imgs_num):
			for k in range(mc_sample // 2):
				print("+ : " + str(multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + k))
				print("- : " + str(multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + (mc_sample // 2) + k))
		print(delta * (mc_sample // 2) * (1 + mc_sample // 2))

grad_test()