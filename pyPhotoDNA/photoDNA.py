#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import base64
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


libPath = r'C:\Users\sungwoo\Downloads\phash_multi_target_resetToSingel\ImageNet_results\pyPhotoDNA\PhotoDNAx64.dll'

def generatePhotoDNAHash(imagePath):

	imageFile = Image.open(imagePath, 'r')
	# if imageFile.mode != 'RGB':
	# 	imageFile = imageFile.convert(mode = 'RGB')
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





h1 = generatePhotoDNAHash(r'C:\Users\sungwoo\Downloads\phash_multi_target_resetToSingel\ImageNet_results\targetImage\id000.png')
h2 = generatePhotoDNAHash(r'C:\Users\sungwoo\Downloads\phash_multi_target_resetToSingel\ImageNet_results\ImageNet_sorted3\id00006.png')

def PhotoDNA_Distance(h1, h2):
	distance = 0
	if len(h1) != 144:
		panic("h1 wrong length")
	
	if len(h2) != 144:
		panic("h2 wrong length")
	

	for i in range(len(h1)):
		distance += abs(h1[i] - h2[i])
	return distance

print(PhotoDNA_Distance(h1, h2))