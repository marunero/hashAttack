

import os
import sys
import argparse
import numpy as np
import random
import cv2
import torch
from matplotlib import pyplot as plt
import datetime


from setup_image_hash import ImageNet, ImageNet_Hash
from attack_hash import hash_attack

# tensorflow
import tensorflow.compat.v1 as tf

# image
from PIL import Image

# hash
import imagehash
import pdqhash

# perceptual similarity
from lpips_tensorflow.lpips_tf import lpips

def gen_image(arr):
    fig = np.around((arr) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img
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
        data, model = ImageNet(), ImageNet_Hash(args['targeted'])
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

            attack = hash_attack(sess, model, args['batch_size'], args['targeted'], args['learning_rate'], args['binary_steps'], args['max_iteration'], args['print_unit'], args['init_const'], args['use_resize'], args['resize_size'], args['use_grayscale'], args['adam_beta1'], args['adam_beta2'], args['mc_sample'], args['multi'], args['optimizer'], args['hash'], args['distance_metric'], input_images.shape[1], input_images.shape[2], input_images.shape[3], target_image.shape[0], target_image.shape[1], target_image.shape[2])



            modifier, loss_x, loss_y = attack.attack_batch(input_images, target_image)

            gen_image(input_images[0] + modifier).save("test.png")

            plt.plot(loss_x, loss_y)
            plt.savefig(str(i) + '_' + str(args['learning_rate']) + '.png')
            plt.clf()





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_idx", type=int, default=0, help="start index of image folder")
    parser.add_argument("-n", "--numimg", type=int, default=0, help="number of test images to attack")
    parser.add_argument("-mu", "--multi", type=int, default=1, help="number of images to attack simultaneously")
    parser.add_argument("-mc", "--mc_sample", type=int, default=2, help="number of samples for Monte Carlo")
    parser.add_argument("-t", "--targeted", action='store_true')
    parser.add_argument("-hash", "--hash", choices=["phash", "pdqhash", "photoDNA"], default="phash")
    parser.add_argument("-dist", "--distance_metric", choices=["l2dist", "pdist"], default="l2dist")
    parser.add_argument("--optimizer", choices=["adam", "momentum"], default="adam")

    parser.add_argument("--use_grayscale", action='store_true', help="convert grayscale image")
    parser.add_argument("--use_resize", action='store_true', help="resize image")
    parser.add_argument("--resize_size", type=int, default=64, help="size of resized modifier")
    parser.add_argument("-p", "--print_unit", type=int, default=1, help="print objs every print_unit iterations")
    parser.add_argument("-c", "--init_const", type=float, default=0.0001)
    parser.add_argument("-bi", "--binary_steps", type=int, default=1)

    
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-mi", "--max_iteration", type=int, default=1000)

    parser.add_argument("--gpu", "--gpu_machine", default="0")

    parser.add_argument("--seed", type=int, default=1359)
    args = vars(parser.parse_args())

    
    random.seed(args['seed'])
    np.random.seed(args['seed'])

    
    print(args)
    main(args)

