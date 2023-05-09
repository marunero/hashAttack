## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.
import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
import random
import time
import torch
import datetime

from matplotlib import pyplot as plt

from setup_imagenet_hash import ImageNet, ImageNet_HashModel
from l2_attack_black import BlackBoxL2

from PIL import Image
import imagehash
import robusthash
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pylab as pylab
import scipy.fftpack
from skimage import transform
import cv2
from skimage import exposure

from lpips_tensorflow.lpips_tf import lpips
import imagehash
import pdqhash

def show(img, name="output.png"):
    """
    Show MNSIT digits in the console.
    """
    fig = np.around((img+0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic = pic.convert('RGB')
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    

def generate_data(data, targeted=True, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    gray_inputs = []
    for i in range(len(data.test_data)):
        inputs.append(data.test_data[i])
        true_ids.append(i)
        gray_inputs.append(data.test_data_gray[i])

        if targeted:
            targets.append(data.target_data[i])
        else:
            targets.append(data.test_data_gray[i])


    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)
    gray_inputs = np.array(gray_inputs)

    return inputs, targets, labels, true_ids, gray_inputs

def gen_image(arr):
    # two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    # img = Image.fromarray(np.uint8(arr * 255))
    # img = Image.fromarray(two_d)

    fig = np.around((arr+0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img




def main(args):
    # config = tf.ConfigProto(device_count={"CPU": 2}, # limit to num_cpu_core CPU usage
    #             inter_op_parallelism_threads = 1, 
    #             intra_op_parallelism_threads = 4,
    #             log_device_placement=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']  # "0,1,2,3".
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    #Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    #Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        use_log = not args['use_zvalue']
        is_inception = args['dataset'] == "imagenet"
        # load network
        print('Loading model', args['dataset'])
        if args['dataset'] == "mnist":
            data, model = MNIST(), MNIST_HashModel()
        # elif args['dataset'] == "maladv":
        #     data, model = MalAdv(), MalAdv_HashModel()
        # elif args['dataset'] == "face":
        #     data, model = Face(), Face_HashModel(args['hash'], args['bits'], args['factor'])
        elif args['dataset'] == "imagenet":
            data, model = ImageNet(), ImageNet_HashModel(args['hash'], args['bits'], args['factor'])
   
        if args['numimg'] == 0:
            args['numimg'] = len(data.test_labels) - args['start_idx']
        print('Using', args['numimg'], 'test images')
        # load attack module
        # if args['attack'] == "white":
        #     # batch size 1, optimize on 1 image at a time, rather than optimizing images jointly
        #     # attack = CarliniL2(sess, model, batch_size=1, max_iterations=args['maxiter'],
        #     #                    print_every=args['print_every'],
        #     #                    early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate=args['lr'],
        #     #                    initial_const=args['init_const'],
        #     #                    binary_search_steps=args['binary_steps'], targeted=not args['untargeted'],
        #     #                    use_log=use_log,
        #     #                    adam_beta1=args['adam_beta1'], adam_beta2=args['adam_beta2'])
        #     print('white')
        # else:
        #     # batch size 128, optimize on 128 coordinates of a single image
        #     attack = BlackBoxL2(sess, model, batch_size=args['batch'], max_iterations=args['maxiter'],
        #                         print_every=args['print_every'],
        #                         early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate=args['lr'],
        #                         initial_const=args['init_const'],
        #                         binary_search_steps=args['binary_steps'], targeted=not args['untargeted'],
        #                         use_log=use_log, use_tanh=args['use_tanh'],
        #                         use_resize=args['use_resize'], adam_beta1=args['adam_beta1'],
        #                         adam_beta2=args['adam_beta2'], reset_adam_after_found=args['reset_adam'],
        #                         solver=args['solver'], save_ckpts=args['save_ckpts'], load_checkpoint=args['load_ckpt'],
        #                         start_iter=args['start_iter'],
        #                         init_size=args['init_size'], use_importance=not args['uniform'], method=args['method'], dct=args['dct'], dist_metrics=args['dist_metrics'], htype=args["htype"], height=288, width=288, channels=1)

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        print('Generate data')
        
        

        all_inputs, all_targets, all_labels, all_true_ids, all_gray_inputs = generate_data(data, targeted = not args['untargeted'], 
                                                                          inception=is_inception)
                                                                          
        print('Done...')
        os.system("mkdir -p {}/{}".format(args['save'], args['dataset']))
        img_no = args["start_idx"]


        total_success = 0
        # l2_total = 0.0
        l2_total = 0.0
        pdistance1_total = 0.0
        hash_total = 0.0
        hash_total2 = 0
      
        l2_total2 = 0
        pdistance_total = 0


        # print('testing for phash differences for %s dataset' % args['dataset'])
        differences = 0
        total = 0

        total_success_iterations = 0
        total_iterations = 0

        # start_idxs = [6, 38, 48]

        total_time = 0
    
        for i in range(args['start_idx'], args['start_idx'] + args['numimg']):
        # for i in start_idxs:
            torch.cuda.empty_cache()  
            print('for image id ', all_true_ids[i])

            multi = args['multi']
            gray_inputs = all_gray_inputs[i]
            
            if args['use_grayscale']:
                multi_inputs = all_gray_inputs[i:i + multi]
                inputs = all_gray_inputs[i]
            else:
                multi_inputs = all_inputs[i:i + multi]
                inputs = all_inputs[i]

            target_hash_inputs = all_targets[i]

            if args['untargeted'] == False:
                if args['htype'] == "phash":
                    input_img = gen_image(inputs)
                    target_img = gen_image(target_hash_inputs)

                    h_d = imagehash.phash(input_img) - imagehash.phash(target_img)
                    print("input[0] and target phash difference = ", h_d)
                elif args['htype'] == "pdqhash":
                    np_input = np.array(gen_image(inputs))
                    cv2_img1 = cv2.cvtColor(np_input, cv2.COLOR_RGB2BGR)
                    h1, q1 = pdqhash.compute(cv2_img1)

                    np_target = np.array(gen_image(target_hash_inputs))
                    cv2_img2 = cv2.cvtColor(np_target, cv2.COLOR_RGB2BGR)
                    h2, q2 = pdqhash.compute(cv2_img2)

                    h_d = ((h1 != h2) * 1).sum()
                    print("input[0] and target pdqhash difference = ", h_d)
                
            
            
            # print(min_hash)
            # print(imagehash.phash(gen_image(all_gray_inputs[min_hash_i])))
            print('each inputs shape ', inputs.shape)
            print('multi inputs shape ', multi_inputs.shape)
            print('each target inputs shape ', target_hash_inputs.shape)
            # if len(gray_inputs.shape) == 4:
            #     gray_inputs = gray_inputs[0]

            img_no += 1

            attack = BlackBoxL2(sess, model, batch_size=args['batch'], max_iterations=args['maxiter'],
                                print_every=args['print_every'],
                                early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate=args['lr'],
                                initial_const=args['init_const'],
                                binary_search_steps=args['binary_steps'], targeted=not args['untargeted'],
                                use_log=use_log, use_tanh=args['use_tanh'],
                                use_resize=args['use_resize'], use_grayscale=args['use_grayscale'], adam_beta1=args['adam_beta1'],
                                adam_beta2=args['adam_beta2'], reset_adam_after_found=args['reset_adam'],
                                solver=args['solver'], attack = args['attack'], save_ckpts=args['save_ckpts'], load_checkpoint=args['load_ckpt'],
                                start_iter=args['start_iter'],
                                init_size=args['init_size'], use_importance=not args['uniform'], method=args['method'], dct=args['dct'], dist_metrics=args['dist_metrics'], htype=args["htype"], height=gray_inputs.shape[0], width=gray_inputs.shape[1], channels=multi_inputs.shape[3], theight=target_hash_inputs.shape[0], twidth=target_hash_inputs.shape[1], tchannels=target_hash_inputs.shape[2], multi_imgs_num = multi, mc_sample = args['mc_sample'])

            print(multi_inputs.shape[3])

            inputs = inputs.reshape((1, ) + inputs.shape)
            target_hash_inputs = target_hash_inputs.reshape((1, ) + target_hash_inputs.shape)

            timestart = time.time()
            
            

            if args['untargeted'] == True:
                targetHashimg = inputs
            else:
                targetHashimg = target_hash_inputs
            
            adv, adv_sec, const, L3, adv_current, first_iteration, nimg, modifier, loss_x, loss_y = attack.attack_batch(inputs, multi_inputs, targetHashimg, i)
            

            # print(adv.shape) = (644, 400, 1)
            # print(adv_current.shape) = (644, 400, 1)
            #print(first_iteration) == (19)

            timeend = time.time()
            print("finish attack adv shape ", adv.shape)
            if type(const) is list:
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            if len(adv_current.shape) == 3:
                adv_current = adv_current.reshape((1,)+ adv_current.shape)

            # l2 distances
            l2_distortion_direct = np.sum((adv - gray_inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv ', l2_distortion_direct)
            l2_distortion_current = np.sum((adv_current - gray_inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv_current ', l2_distortion_current)
            if len(inputs.shape) == 4:
                a,b,c = gray_inputs[0].shape
                print("normalized a,b,c", a, b, c)
                l2_distortion_normalized = l2_distortion_direct / (a*b*c)**0.5
                l2_distortion_current_normalized = l2_distortion_current / (a*b*c)**0.5

            if len(gray_inputs.shape) == 4:
                stacked_gray_inputs = gray_inputs[0]
                stacked_adv = adv[0]
                stacked_adv_current = adv_current[0]
            else:
                stacked_gray_inputs = gray_inputs
                stacked_adv = adv
                stacked_adv_current = adv_current


            image0_ph = tf.placeholder(tf.float32)
            image1_ph = tf.placeholder(tf.float32)
            distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='alex')
           
            stacked_gray_inputs =  np.asarray(np.dstack((stacked_gray_inputs, stacked_gray_inputs, stacked_gray_inputs)))
            stacked_adv =  np.asarray(np.dstack((stacked_adv, stacked_adv, stacked_adv)))
            stacked_adv_current =  np.asarray(np.dstack((stacked_adv_current, stacked_adv_current, stacked_adv_current)))
            if len(stacked_gray_inputs.shape) == 3:
                stacked_gray_inputs = stacked_gray_inputs.reshape((1,) + stacked_gray_inputs.shape)
                stacked_adv = stacked_adv.reshape((1,) + stacked_adv.shape)
                stacked_adv_current = stacked_adv_current.reshape((1,) + stacked_adv_current.shape)

            print(stacked_adv.shape)
            print(stacked_adv_current.shape)
            # perceptual distances
            with tf.Session(config=config) as session:
                distance1 = session.run(distance_t, feed_dict={image0_ph: (stacked_gray_inputs+0.5), image1_ph: (stacked_adv+0.5)})
                distance2 = session.run(distance_t, feed_dict={image0_ph: (stacked_gray_inputs + 0.5), image1_ph: (stacked_adv_current+0.5)})

            distance1_normalized = distance1[0] / (a*b*c)**0.5
            distance2_normalized = distance2[0] / (a*b*c)**0.5
            success = False
            print("perceptual metrics distance between adv and img", distance1[0])
            print("perceptual metrics distance between adv_current and img", distance2[0])
            print("normalized perceptual metrics distance between adv and img", distance1_normalized)
            print("normalized perceptual metrics distance between adv_current and img", distance2_normalized)

            print(adv.shape)
            inputs_img = gen_image(gray_inputs)

            adv = adv[0]

            adv_img = gen_image(adv)
            target_hash_img = gen_image(targetHashimg)

            if args["htype"] == "phash":
                if args['untargeted'] == True:
                    hash_differences = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                else:
                    hash_differences = imagehash.phash(target_hash_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                print('perceptual hash difference is ', hash_differences)

            elif args["htype"] == "blockhash":
                if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                    im_original = inputs_img.convert('RGB')
                    im_adver = adv_img.convert('RGB')
                    hash_differences = sum(1 for i, j in zip(robusthash.blockhash(im_original), robusthash.blockhash(im_adver)) if i != j)
                print('robust hash differences', hash_differences)

            elif args["htype"] == "pdqhash":
                np_adv = np.array(adv_img)
                cv2_img1 = cv2.cvtColor(np_adv, cv2.COLOR_RGB2BGR)
                h1, q1 = pdqhash.compute(cv2_img1)

                if args['untargeted'] == True:
                    np_ori = np.array(target_hash_img)
                else:
                    np_ori = np.array(inputs_img)

                cv2_img2 = cv2.cvtColor(np_ori, cv2.COLOR_RGB2BGR)
                h2, q2 = pdqhash.compute(cv2_img2)

                hash_differences = ((h1 != h2) * 1).sum()
                print('pdq hash difference is ', hash_differences)
                
           
            # inputs_arr = np.asarray(inputs_img) / 255.0 
            # adv_arr = np.asarray(adv_img) / 255.0 
            # l2_distortion = np.sum((inputs_arr - adv_arr) ** 2) ** 0.5

            if L3 == True: 
                # if args["htype"] == "phash":
                #     hash_differences = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                #     print('perceptual hash difference is ', hash_differences)
                # elif args["htype"] == "blockhash":
                #     if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                #         im_original = inputs_img.convert('RGB')
                #         im_adver = adv_img.convert('RGB')
                #     hash_differences = sum(1 for i, j in zip(robusthash.blockhash_even(im_original), robusthash.blockhash_even(im_adver)) if i != j)
                #     print('robust hash differences', hash_differences)


                if args['untargeted']:
                    if hash_differences >= args['hash']:
                        #print('hash difference threshold ', args['hash'])
                        success = True
                else:
                    if hash_differences <= 0:
                        success = True

                if success:
                    total_success += 1
                    l2_total += l2_distortion_direct
                    pdistance1_total += distance1[0]
                    hash_total += hash_differences
                    hash_total2 += hash_differences
                    l2_total2 += l2_distortion_direct
                    pdistance_total += distance1[0]
                    total_success_iterations += first_iteration
                    total_iterations += first_iteration
                    if args['untargeted'] == True:
                        suffix = "id{}_differ{}_{}_pdistance{}".format(all_true_ids[i], hash_differences, success, distance1[0])
                    else:
                        suffix = "id{}_differ{}_{}_pdistance{}_targetImageId{}".format(all_true_ids[i], hash_differences, success, distance1[0], all_true_ids[i])
                    print("Saving to", suffix)
                    
                    if args['translateRGB']:
                        differ = adv - gray_inputs

                        adv_rgb = inputs
                        adv_rgb = adv_rgb + differ
                        adv_rgb = np.clip(adv_rgb, -0.5, 0.5)
                        differ = np.clip(differ, -0.5, 0.5)

                        show(adv_rgb - inputs, "{}/{}_advNoise_{}.png".format(args['save'], img_no,suffix))
                        show(adv, "{}/{}_advRGB_{}.png".format(args['save'], img_no,suffix))
                        show(adv_sec, "{}/{}_adv2RGB_{}.png".format(args['save'], img_no,suffix))
                        # show(targetHashimg, "{}/{}_advTarget_{}.png".format(args['save'], img_no,suffix))
                        # show(adv, "{}/{}_adversarial_{}.png".format(args['save'], img_no, suffix))
                        # show(differ, "{}/{}_differ_{}.png".format(args['save'], img_no,suffix))
                        # show(gray_inputs, "{}/{}_original_{}.png".format(args['save'], img_no,suffix))
                    else:
                        # show(gray_inputs, "{}/{}_original_{}.png".format(args['save'], img_no,suffix))
                        show(adv, "{}/{}_adversarial_{}.png".format(args['save'], img_no, suffix))

                    # for name saving purposes, 2nd calculation done
                    print("[STATS][L1] total = {}, id = {}, time = {:.3f}, success = {}, const = {:.6f}, hash_avg={:.5f}, distortion = {:.5f}, success_rate = {:.3f}, l2_avg={:.5f}, p_avg={}, iteration_avg={}"
                    .format(img_no, all_true_ids[i], timeend - timestart, success, const, 0 if total_success == 0 else hash_total / total_success, l2_distortion_direct, total_success / float(img_no), 0 if total_success == 0 else l2_total / total_success, 0 if total_success == 0 else pdistance1_total/ total_success
                    , 0 if total_success == 0 else total_success_iterations/ total_success))
                    sys.stdout.flush()
            else:
                
                adv_current_img = gen_image(adv_current)
                print("unsuccessful")

                if args["htype"] == "phash":

                    if args['untargeted'] == True:
                        hash_differences_current = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_current_img, args['bits'], args['factor'])
                    else:
                        hash_differences_current = imagehash.phash(target_hash_img, args['bits'], args['factor']) - imagehash.phash(adv_current_img, args['bits'], args['factor'])
                    
                    print('perceptual hash difference is ', hash_differences_current)
                    
                elif args["htype"] == "blockhash":
                    if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                        im_original = inputs_img.convert('RGB')
                        im_adver_current = adv_current_img.convert('RGB')
                    hash_differences_current = sum(1 for i, j in zip(robusthash.blockhash(im_original), robusthash.blockhash(im_adver_current)) if i != j)
                    print('robust hash differences', hash_differences_current)

                elif args["htype"] == "pdqhash":
                    np_adv = np.array(adv_current_img)
                    cv2_img1 = cv2.cvtColor(np_adv, cv2.COLOR_RGB2BGR)
                    h1, q1 = pdqhash.compute(cv2_img1)

                    if args['untargeted'] == True:
                        np_ori = np.array(target_hash_img)
                    else:
                        np_ori = np.array(inputs_img)

                    cv2_img2 = cv2.cvtColor(np_ori, cv2.COLOR_RGB2BGR)
                    h2, q2 = pdqhash.compute(cv2_img2)

                    hash_differences_current = ((h1 != h2) * 1).sum()
                    print('pdq hash difference is ', hash_differences_current)


                total_iterations += first_iteration
                hash_total2 += hash_differences_current
                l2_total2 += l2_distortion_current
                pdistance_total +=  distance2[0]
                print("Failed attacks!")
                # for name saving purposes, 2nd calculation done
                suffix_current = "l2_{:.2f}_pdist{}_diff{}_success={}_time{}".format(l2_distortion_current, distance2[0], hash_differences_current, success, timeend - timestart)
                

                differ = adv_current - gray_inputs
                
                adv_rgb = inputs
                adv_rgb = adv_rgb + differ
                adv_rgb = np.clip(adv_rgb, -0.5, 0.5)
                differ = np.clip(differ, -0.5, 0.5)

                show(adv_current, "{}/id{}_adv_current_{}.png".format(args['save'],  i, suffix_current))
                
                # if args['translateRGB']:
                    
                
                print('saving for failed attack current', suffix_current)
                sys.stdout.flush()
            
            print("overall average hash ", hash_total2 / (img_no - args["start_idx"]))
            print("overall average l2 ", l2_total2 /(img_no-args["start_idx"]))
            print("overal average perceptual distance ", pdistance_total / (img_no-args["start_idx"]))
            print("overal average iterations ", total_iterations / (img_no-args["start_idx"]))

            # print(loss_x)
            # print(loss_y)

            for j in range(int(len(loss_x) / 2)):
                plot_suffix = "result/id" + str(i) + "_binery_step" + str(j + 1)
  
                plt.figure()

                plt.plot(loss_x[j * 2], loss_y[j * 2], label='sum of 1 and 2')
                plt.legend(loc='best')
                plt.plot(loss_x[j * 2 + 1], loss_y[j * 2 + 1], label='hash loss - imgs')
                plt.legend(loc='best')

                
                now = datetime.datetime.now()
                current_time = now.strftime("%m_%d_%H_%M")

                plt.savefig(plot_suffix + current_time + ".png")
                plt.clf()
            plt.close('all')    

            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet", "maladv", "face", "random"], default="imagenet")
    parser.add_argument("-s", "--save", default="./saved_results")
    parser.add_argument("-n", "--numimg", type=int, default=0, help="number of test images to attack")
    parser.add_argument("-mu", "--multi", type=int, default=1, help="number of images to attack simultaneously")
    parser.add_argument("-mc", "--mc_sample", type=int, default=2, help="number of images to attack simultaneously")
    parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to use default value")
    parser.add_argument("-p", "--print_every", type=int, default=100, help="print objs every PRINT_EVERY iterations")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=0,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("-b", "--binary_steps", type=int, default=0)
    parser.add_argument("-c", "--init_const", type=float, default=0.0)
    parser.add_argument("-z", "--use_zvalue", action='store_true')
    parser.add_argument("-u", "--untargeted", action='store_true')
    parser.add_argument("-trgb", "--translateRGB", action='store_true')
    parser.add_argument("-r", "--reset_adam", action='store_true', help="reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action='store_true', help="resize image (only works on imagenet!)")
    parser.add_argument("--use_grayscale", action='store_true', help="convert grayscale image")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--solver", choices=["adam", "adam2", "adam2_newton", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument("--attack", choices=["basic", "transform_rot", "transform_crop", "transform_scale", "transform_ensemble"], default="basic")
    parser.add_argument("--save_ckpts", default="", help="path to save checkpoint file")
    parser.add_argument("--load_ckpt", default="", help="path to numpy checkpoint file")
    parser.add_argument("--start_iter", default=0, type=int,
                        help="iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default=64, type=int, help="starting with this size when --use_resize")
    parser.add_argument("--uniform", action='store_true', help="disable importance sampling")
    parser.add_argument("--method", "--transform_method",  default='linear')
    parser.add_argument("--gpu", "--gpu_machine", default="0")
    parser.add_argument("--hash", "--hashbits", type=int, default=6)
    parser.add_argument("--batch", "--batchsize", type=int, default=128)
    parser.add_argument("--start_idx", "--start image index", type=int, default=0)
    parser.add_argument("--dct", "--if using dct compression", action='store_true')
    # parser.add_argument("--num_rand_vec", "--random number of vectors like batch", type=int, default=1)
    parser.add_argument("--lr", "--learning rate", type=float, default=0.1)
    parser.add_argument("--transform", "--basic transormation", default="centrol_crop")
    parser.add_argument('--dist_metrics', "--distance metrics to use", choices=["l2dist", "pdist"], default="l2dist")
    parser.add_argument("--bits", "--hash_string_length", type=int, default=8)
    parser.add_argument("--factor", "--hash_string_factor", type=int, default=4)
    parser.add_argument("--maximize", "--if_plus_or_minus", choices=["plus", "minus"], default="minus")
    parser.add_argument("-ht", "--htype", choices=["phash", "blockhash", "pdqhash", "photoDNA"], default="phash")
    parser.add_argument("-ra", "--ratio", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1307)
    args = vars(parser.parse_args())

    
    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])


    # add some additional parameters
    # learning rate
    #args['lr'] = 1e-2
    args['inception'] = False
    args['use_tanh'] = True
    # args['use_resize'] = False
    if args['maxiter'] == 0:
        # for hash=10,20 - 2000, 30-3000
        if args['dataset'] == "imagenet":
            if args['untargeted']:
                # for imagenet resize
                args['maxiter'] = 2000
            else:
                args['maxiter'] = 50000
        elif args['dataset'] == "mnist":
            args['maxiter'] = 3000
        else:
            args['maxiter'] = 2000
    if args['init_const'] == 0.0:
        if args['binary_steps'] != 0:
            args['init_const'] = 100
        else:
            args['init_const'] = 1
    if args['binary_steps'] == 0:
        args['binary_steps'] = 1
    # set up some parameters based on datasets
    # if args['dataset'] == "imagenet":
    #     args['inception'] = True
    #     args['lr'] = 1e-3
    #     # args['use_resize'] = True
    #     # args['save_ckpts'] = True
    # if args['dataset'] == "maladv" or args['dataset'] == "face":
    #     args['lr'] = 2e-3
    # for mnist, using tanh causes gradient to vanish
    if args['dataset'] == "mnist":
        args['use_tanh'] = True
    # when init_const is not specified, use a reasonable default
    # if args['init_const'] == 0.0:
    #     if args['binary_search']:
    #         args['init_const'] = 0.01
    #     else:
    #         args['init_const'] = 0.5
    print(args)
    main(args)