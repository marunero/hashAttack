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

def etc():
	img1 = Image.open('targetImage/id000.png')


	min_hash = -1
	min_img_name = ""
	
	
	f1, m1, d1 = phash(img1)

	i1 = Image.open("ImageNet_sorted3/id000.png")
	i2 = Image.open("ImageNet_sorted3/id001.png")

	print(imagehash.phash(i1) - imagehash.phash(img1))
	print(imagehash.phash(i2) - imagehash.phash(img1))

	f2, m2, d2 = phash(i1)
	f3, m3, d3 = phash(i2)


	
	print((np.power((f2 - m2) * ((d2 != d1) * 1), 2).sum()) ** 0.5)
	print((np.power((f3 - m3) * ((d3 != d1) * 1), 2).sum()) ** 0.5)


	# source_folder = 'C:/Users/sungwoo/Downloads/imgdata'
	# destination_folder = 'C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/targetImage'
	# filename = min_img_name

	# shutil.copy(os.path.join(source_folder, filename), os.path.join(destination_folder, file_name))



def resizeImage():
	# Load the image
	image = Image.open('ImageNet_sorted3/id000.png')

	# Resize the image
	new_size = (500, 375)
	resized_image = image.resize(new_size)

	# Save the resized image
	resized_image.save('ImageNet_sorted3/resized_image.png')

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


def transfer_ImageNet():
	adv = Image.open("test/1_advRGB_id0_differ0_True_pdistance0.2655538320541382_targetImageId0.png").convert("L")
	noise = np.load("np/best_modifier_img0_2.npy")

	gen_image(noise).show()

	o1 = Image.open("ImageNet_sorted3/id000.png").convert("L")

	i1 = np.array(o1)

	gray_input1 = resize(i1, (i1.shape[0], i1.shape[1], 1), anti_aliasing=True)
	gray_input1 -= 0.5
	

	gray_input1 = np.arctanh(gray_input1*1.999999) 
	gray_input1 = gray_input1.astype(np.float32)

            

	timg1 = tf.Variable(gray_input1, dtype=tf.float32)
	scaled_modifier = tf.image.resize_images(noise, [375, 500])

	
	newimg1 = tf.tanh(scaled_modifier[0] + timg1) / 2
	print(imagehash.phash(gen_image(newimg1)) - imagehash.phash(adv))

	o2 = Image.open("ImageNet_sorted3/id001.png").convert("L")

	i2 = np.array(o2)

	gray_input2 = resize(i2, (i2.shape[0], i2.shape[1], 1), anti_aliasing=True)
	gray_input2 -= 0.5
	

	gray_input2 = np.arctanh(gray_input2*1.999999) 
	gray_input2 = gray_input2.astype(np.float32)

	timg2 = tf.Variable(gray_input2, dtype=tf.float32)
	scaled_modifier = tf.image.resize_images(noise, [375, 500])

	
	newimg2 = tf.tanh(scaled_modifier[0] + timg2) / 2
	print(imagehash.phash(gen_image(newimg2)) - imagehash.phash(adv))


	ImageNet_path = "C:/Users/sungwoo/Downloads/imgdata"
	file_list = os.listdir(ImageNet_path)

	target_hash = imagehash.phash(Image.open("targetImage/id000.png"))

	result = []
	s = 0
	c = []
	a = []
	away = 0
	for i, file_name in enumerate(file_list):
		if i >= 2000:
			break
		img = Image.open(os.path.join(ImageNet_path, file_name))
		original_hash_differ = imagehash.phash(img) - target_hash

		np_img = np.array(img)
		gray_input = resize(np_img, (np_img.shape[0], np_img.shape[1], 1), anti_aliasing = True)
		gray_input -= 0.5
		gray_input = np.arctanh(gray_input*1.999999) 
		gray_input = gray_input.astype(np.float32)
		timg = tf.Variable(gray_input, dtype=tf.float32)
		
		scaled_modifier = tf.image.resize_images(noise, [np_img.shape[0], np_img.shape[1]])
		newimg = tf.tanh(scaled_modifier[0] + timg) / 2

		new_hash_differ = imagehash.phash(gen_image(newimg)) - target_hash

		print(original_hash_differ, new_hash_differ, original_hash_differ - new_hash_differ)
		if (new_hash_differ == 0):
			c.append(file_name)	
			continue
		
		if (original_hash_differ > new_hash_differ):
			if new_hash_differ < 10:
				result.append(file_name)
			s += original_hash_differ - new_hash_differ
		elif (original_hash_differ < new_hash_differ):
			a.append(file_name)
			away += original_hash_differ - new_hash_differ
	print(result)

	print(len(result))
	print(s / len(result))
	print(away / len(a))
	print(c)
	print(len(a), len(result))

	


def test():
	


	a1 = Image.open("test/1_advRGB_id0_differ0_True_pdistance0.2655538320541382_targetImageId0.png").convert("L")
	a2 = Image.open("test/1_adv2RGB_id0_differ0_True_pdistance0.2655538320541382_targetImageId0.png").convert("L")

	o1 = Image.open("ImageNet_sorted3/id000.png").convert("L")
	o2 = Image.open("ImageNet_sorted3/id001.png").convert("L")

	na1 = np.array(a1)
	na1 = resize(na1, (na1.shape[0], na1.shape[1], 1), anti_aliasing=True)
	na1 -= 0.5
	no1 = np.array(o1)
	no1 = resize(no1, (no1.shape[0], no1.shape[1], 1), anti_aliasing=True)
	no1 -= 0.5

	na2 = np.array(a2)
	na2 = resize(na2, (na2.shape[0], na2.shape[1], 1), anti_aliasing=True)
	na2 -= 0.5
	no2 = np.array(o2)
	no2 = resize(no2, (no2.shape[0], no2.shape[1], 1), anti_aliasing=True)
	no2 -= 0.5

	noise1 = na1 - no1
	mask = (noise1 > 0.5)
	noise1[mask] -= 1
	mask = (noise1 < -0.5)
	noise1[mask] += 1

	noise2 = na2 - no2 
	mask = (noise2 > 0.5)
	noise2[mask] -= 1
	mask = (noise2 < -0.5)
	noise2[mask] += 1

	a = []
	for i in range(9):
		a.append((i + 1) / 100)

	for i in range(9):
		alpha = a[i]

		avg_noise = noise1 * alpha + noise2 * (1 - alpha) 

		# new_a = avg_noise + no2
		# mask = (new_a > 0.5)
		# new_a[mask] -= 1
		# mask = (new_a < -0.5)
		# new_a[mask] += 1

		new_a1 = np.clip(avg_noise + no1, -0.5, 0.5)
		new_a2 = np.clip(avg_noise + no2, -0.5, 0.5)


		# gen_image(new_a2).show()
		d1 = imagehash.phash(gen_image(new_a1)) - imagehash.phash(a2)
		d2 = imagehash.phash(gen_image(new_a2)) - imagehash.phash(a2)

		print(d1, d2)


	# noise = np.load("np/best_modifier_img0.npy")


	# i = np.array(o1)
	# gray_input = resize(i, (i.shape[0], i.shape[1], 1), anti_aliasing=True)
	# gray_input -= 0.5
	

	# gray_input = np.arctanh(gray_input*1.999999) 
	# gray_input = gray_input.astype(np.float32)

            

	# timg = tf.Variable(gray_input, dtype=tf.float32)
	# scaled_modifier = tf.image.resize_images(noise, [375, 500])

	
	# newimg = np.clip(scaled_modifier[0] + timg, -0.5, 0.5)
	# gen_image(newimg).show()
	# print(imagehash.phash(gen_image(newimg)) - imagehash.phash(i1))








	# i3 = Image.open("test/1_advNoise_id0_differ0_True_pdistance0.2655538320541382_targetImageId0.png").convert("L")

	# target = Image.open("targetImage/id000.png").convert("L")

	# h1 = imagehash.phash(i1)
	# h2 = imagehash.phash(i2)
	# h3 = imagehash.phash(target)

	# print(h1 - h3)
	# print(h2 - h3)


	# noise = np.load("np/best_modifier_img0.npy")
	# new_noise = tf.image.resize_images(noise, [375, 500])

	# new_noise = new_noise / np.max(new_noise) / 2
	

	# i1 = Image.open("ImageNet_sorted3/id000.png").convert("L")
	# ni1 = np.array(i1)
	# ni1 = resize(ni1, (ni1.shape[0], ni1.shape[1], 1), anti_aliasing=True)
	# ni1 -= 0.5
	
	# ni1 = np.arctanh(ni1*1.999999)
	# ni1 = ni1.astype(np.float32)
	
	
	# s = tf.tanh(new_noise + ni1)
	# adv_image = gen_image(s)
	# adv_image.show()
	# # gen_image(new_noise).show()
	# print(h1 - imagehash.phash(adv_image))

	# i2 = Image.open("ImageNet_sorted3/id001.png").convert("L")
	# ni2 = np.array(i2)
	# ni2 = resize(ni2, (ni2.shape[0], ni2.shape[1], 1), anti_aliasing=True)
	# ni2 -= 0.5

	# ni2 = np.arctanh(ni2*1.999999)
	# ni2 = ni2.astype(np.float32)
	
	# s = tf.tanh(new_noise + ni2)
	# adv_image = gen_image(np.clip(s, -0.5, 0.5))
	# adv_image.show()
	# # gen_image(new_noise).show()
	# print(h2 - imagehash.phash(adv_image))


def printHash():
	# f1 = "ImageNet_sorted3/id001.png"
	# f2 = "ImageNet_sorted3/id000.png"

	img1 = Image.open("ImageNet_sorted3/id000.png")
	img2 = Image.open("ImageNet_sorted3/id001.png")
	img3 = Image.open("targetImage/id000.png")

	# target
	f1 = "target/1_advTarget_id0_differ0_True_pdistance0.2649911046028137_targetImageId0.png"
	# input 1 grayscale + noise
	f2 = "target/1_advRGB_id0_differ0_True_pdistance0.2649911046028137_targetImageId0.png"
	# noise
	f3 = "target/1_advNoise_id0_differ0_True_pdistance0.2649911046028137_targetImageId0.png"

	img1_gray = img1.convert("L")
	img1_gray = np.array(img1_gray)
	img1_gray = resize(img1_gray, (img1_gray.shape[0], img1_gray.shape[1], 1), anti_aliasing=True)

	noise = Image.open(f3).convert("L")
	noise = np.array(noise)
	noise = resize(noise, (noise.shape[0], noise.shape[1], 1), anti_aliasing=True)

	
	add_hash = imagehash.phash(gen_image(img1_gray + noise))
	adv_img1 = Image.open(f2)
	adv_img1_hash = imagehash.phash(adv_img1)

	print(add_hash - adv_img1_hash)

	target = img3.convert("L")
	target_hash = imagehash.phash(target)
	print(target_hash - adv_img1_hash)


	img2_gray = img2.convert("L")
	img2_gray = np.array(img2_gray)
	img2_gray = resize(img2_gray, (img2_gray.shape[0], img2_gray.shape[1], 1), anti_aliasing=True)

	add_img2_hash = imagehash.phash(gen_image(img2_gray + noise))
	
	print(add_hash - add_img2_hash)
	print(target_hash - add_img2_hash)

	# f1 = "C:/Users/sungwoo/Downloads/data/id10.png"
	# f2 = "C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/basic20/10_advRGB_id9_differ20_True_pdistance0.018991131335496902.png"



	# i1 = Image.open(f1)
	# i2 = Image.open(f2)

	# f1, m1, d1 = phash(i1)
	# f2, m2, d2 = phash(i2)	
	# hash_differ = (np.power((f1 - m1) * ((d1 != d) * 1), 2).sum()) ** 0.5
	# print(hash_differ)

	# print(imagehash.phash(i1) - imagehash.phash(i2))





def testImage():
	f1 = "C:/Users/sungwoo/Downloads/ramen1.png"
	f2 = "C:/Users/sungwoo/Downloads/ramen2.png"
	f3 = "C:/Users/sungwoo/Downloads/ramen3.png"

	# f1 = "C:/Users/sungwoo/Downloads/data/id10.png"
	# f2 = "C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/basic20/10_advRGB_id9_differ20_True_pdistance0.018991131335496902.png"



	i1 = Image.open(f1).convert('L')
	i2 = Image.open(f2).convert('L')
	i3 = Image.open(f3).convert('L')

	h1 = imagehash.phash(i1)
	h2 = imagehash.phash(i2)
	h3 = imagehash.phash(i3)
	print(h1 - h2)
	print(h2 - h3)
	print(h1 - h3)
testImage()

def sortFiles():
	folder_path = "C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/ImageNet_sorted3"
	file_list = os.listdir(folder_path)

	for i, file_name in enumerate(file_list):
		file_path = os.path.join(folder_path, file_name)
		
		imgdata_path = "C:/Users/sungwoo/Downloads/imgdata"

		img_list = os.listdir(imgdata_path)
		
		print(file_name)

		min_hash = -1
		min_img_name = ""
		f1 = file_path
		img1 = Image.open(f1)
		h1 = imagehash.phash(img1)

		for j, img_name in enumerate(img_list):			
			f2 = os.path.join(imgdata_path, img_name)
			img2 = Image.open(f2)

			hash_differ = h1 - imagehash.phash(img2)
			if hash_differ == 0:
				continue
			if min_hash < 0:
				min_hash = hash_differ
				min_img_name = img_name
			else:
				if min_hash > hash_differ:
					min_hash = hash_differ
					min_img_name = img_name

		source_folder = 'C:/Users/sungwoo/Downloads/imgdata'
		destination_folder = 'C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/targetImage'
		filename = min_img_name

		shutil.copy(os.path.join(source_folder, filename), os.path.join(destination_folder, file_name))


def nameSort():
	folder_path = "C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/targetImage"

	file_list = os.listdir(folder_path)

	for i, file_name in enumerate(file_list):
		# new_file_name = "id{:03d}.png".format(i)

		# os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

		source_folder = "C:/Users/sungwoo/Downloads/phash_public-main/ImageNet_results/ImageNet_sorted3"

		h1 = imagehash.phash(Image.open(folder_path + "/" + file_name))
		h2 = imagehash.phash(Image.open(source_folder + "/" + file_name))

		print(h1 - h2)

# printHash()
# target_list = sorted(os.listdir("./targetImage"))
# print(target_list)