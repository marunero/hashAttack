
from __future__ import absolute_import, division, print_function
from torchvision.transforms import ToTensor, ToPILImage

import os
import sys
import numpy as np
import random

from skimage.transform import rescale, resize, downscale_local_mean
import time

from PIL import Image
from torchvision import transforms
import cv2

import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기 
import pdqhash
import imagehash
# photoDNA
import glob
import base64
from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

from tqdm import tqdm


libPath = r'C:\Users\sungwoo\Downloads\hashAttack\pyPhotoDNA\PhotoDNAx64.dll'

mode = "phash"

def differ(h1, h2):
    return h1 - h2


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

def h2str(hash):
    result = ""
    for i in range(len(hash)):
        result += str(hash[i])
    return result

def get_threshold():
    ImageNet_path = "C:/Users/sungwoo/Downloads/data_hashAttack/input/category2_test"
    file_list = os.listdir(ImageNet_path)
    
    hashes = []
    
    print(len(file_list))
    start_time = time.time()

    # count = 100

    for i, file_name in tqdm(enumerate(file_list)):
        # if i > count:
        #     break
        # i += 1
        img = Image.open(os.path.join(ImageNet_path, file_name)).convert('RGB')
        
        # phash
        if mode == "phash":
            # h = imagehash.phash(img)
            h = imagehash.phash(img, hash_size=8)

        # photoDNA
        elif mode == "photoDNA":
            h = generatePhotoDNAHash(img)

        else:
        # elif mode == "PDQ":
            np_img = np.array(img)
            cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            h, q = pdqhash.compute(cv2_img)

        hashes.append(h)


    list_differ = []

    set_differ = {}
    
    status = True
    min_d = -1
    max_d = -1

    print("hash list done")

    for i in tqdm(range(len(hashes))):
        for j in range(i + 1, len(hashes)):
            if mode == "phash":
                dif = differ(hashes[i], hashes[j])
                list_differ.append(dif)
            elif mode == "photoDNA":
                dif = PhotoDNA_Distance(hashes[i], hashes[j])
                if status:
                    min_d = dif
                    max_d = dif
                    status = False
                else:
                    min_d = min(dif, min_d)
                    max_d = max(dif, max_d)
                if dif in set_differ:
                    set_differ[dif] += 1
                else:
                    set_differ[dif] = 1
            elif mode == "PDQ":
                dif = ((hashes[i] != hashes[j]) * 1).sum()
                list_differ.append(dif)
                
            if status:
                min_d = dif
                max_d = dif
                status = False
            else:
                min_d = min(dif, min_d)
                max_d = max(dif, max_d)
            
            if dif in set_differ:
                set_differ[dif] += 1
            else:
                set_differ[dif] = 1

    end_time = time.time()

    # print(set_differ.keys().min())
    sorted_keys = sorted(set_differ.keys())
    print(sorted_keys)

    elapsed_time = end_time - start_time 
    print("총 실행 시간:", elapsed_time, "초")


    print(min_d, max_d)

    if mode == "phash":
        bins = [i for i in range(256)]


        hist, bins = np.histogram(list_differ, bins=bins)

        nonzero_bins = hist.nonzero()
        new_bin_edges = bins[nonzero_bins]
        new_hist = hist[nonzero_bins]

        plt.bar(bins[:-1], hist)
        plt.xticks(new_bin_edges)
        plt.xlim(new_bin_edges[0], new_bin_edges[-1])
        plt.xlabel('distance')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig('phash_256.png', dpi=300)
    elif mode == "photoDNA":
        print(min_d, max_d)


        x = list(set_differ.keys())
        y = list(set_differ.values())
        plt.scatter(x, y, s = 2)

        # 그래프 세부 설정
        plt.xlabel('distance')
        plt.ylabel('frequency')
        plt.show()
    elif mode == "PDQ":
        bins = [i for i in range(257)]
        x = list(set_differ.keys())
        y = list(set_differ.values())
        plt.scatter(x, y, s = 2)

        # 그래프 세부 설정
        plt.xlabel('distance')
        plt.ylabel('frequency')
        plt.show()

# def add_noise_RGB(img, noise):
#     for width in range(img.shape[0]):
#         for height in range(img.shape[1]):

            
def gen_image(arr):
    fig = np.around((arr+0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img

if __name__ == "__main__":
    get_threshold()


    


