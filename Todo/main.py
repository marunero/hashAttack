

import os
import sys
import argparse
import numpy as np
import random
import cv2

from setup_imagenet_hash import ImageNet, ImageNet_Hash

# tensorflow
import tensorflow.compat.v1 as tf

# image
from PIL import Image

# hash
import imagehash
import pdqhash
import robusthash

# perceptual similarity
from lpips_tensorflow.lpips_tf import lpips

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']  # "0,1,2,3".
    
    
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    
    #Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    #Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        
        print('Generating data')
        data, model = ImageNet(), ImageNet_Hash()
        print('Done...')

        print('Using', args['numimg'], 'images')


        img_idx = args["start_idx"]

        for i in range(args['start_idx'], args['start_idx'] + args['numimg']):
            torch.cuda.empty_cache()
            multi = args['multi']

            if args['use_grayscale']:
                input_images = data.input_images_gray[i:i + multi]
            else:
                input_images = data.input_images_rgb[i:i + multi]
            
            target_image = data.target_images_rgb[i]
            

            # print initial hash differnece

            # if args['targeted'] == True:
            #     if args['htype'] == "phash":

            #     elif args['htype'] == "pdqhash":
                    
            #     elif args['htype'] == "photoDNA":
                    
            
            print('input images shape ', input_images.shape)
            print('target image shape ', target_image.shape)

            attack = hash_attack(sess, model)
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_idx", type=int, default=0, help="start index of image folder")
    parser.add_argument("-n", "--numimg", type=int, default=0, help="number of test images to attack")
    parser.add_argument("-mu", "--multi", type=int, default=1, help="number of images to attack simultaneously")
    parser.add_argument("-mc", "--mc_sample", type=int, default=2, help="number of samples for Monte Carlo")
    parser.add_argument("-t", "--targeted", action='store_true')
    parser.add_argument("-h", "--hash", choices=["phash", "pdqhash", "photoDNA"], default="phash")
    parser.add_argument("-p", "--perceptual_metric", choices=["l2dist", "pdist"], default="l2dist")

    parser.add_argument("--use_grayscale", action='store_true', help="convert grayscale image")

    
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("-lr", "--learning rate", type=float, default=0.1)

    parser.add_argument("--gpu", "--gpu_machine", default="0")

    parser.add_argument("--seed", type=int, default=1307)
    args = vars(parser.parse_args())

    
    random.seed(args['seed'])
    np.random.seed(args['seed'])

