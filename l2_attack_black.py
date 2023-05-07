# this is about difference from transformed images, not in loss but monitoring
## l2_attack_black.py -- attack a black-box network optimizing for l_2 distance
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

# Basic transformation + both l2 distances
import sys
import os
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.misc
from numba import jit
import math
import time
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
from PIL import Image
from six.moves import urllib
import imagehash

from utils import load_extractor
BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
LEARNING_RATE = 9e-1     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 0.5      # the initial constant c to pick as a first guess

delta = 0.19999


_URL = 'http://rail.eecs.berkeley.edu/models/lpips'
# import GPUtil
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# DEVICE_ID_LIST = GPUtil.getFirstAvailable()
# DEVICE_ID = DEVICE_ID_LIST[0] 
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
# device = '/gpu:0'
# print('Device ID (unmasked): ' + str(DEVICE_ID))
# print('Device ID (masked): ' + str(0))

def gen_image(arr):
    fig = np.around((arr + 0.5) * 255.0)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)
    return img

def _download(url, output_dir):

    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj, multi_imgs_num, mc_sample):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    
    for i in range(batch_size):
        grad[i] = 0

        for j in range(multi_imgs_num):
            for k in range(mc_sample // 2):
                grad[i] += losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + k]
                grad[i] -= losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + (mc_sample // 2) + k]
        grad[i] /= delta * (mc_sample // 2) * (1 + mc_sample // 2)


        # grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.6
        # grad[i] = (losses[i * 4 + 2] + losses[i * 4 + 3] - losses[i * 4 + 4] - losses[i * 4 + 5]) / (6 * delta)
        # grad[i] += (losses[batch_size * 4 + i * 4 + 2] + losses[batch_size * 4 + i * 4 + 3] - losses[batch_size * 4 + i * 4 + 4] - losses[batch_size * 4 + i * 4 + 5]) / (6 * delta)
        # grad[i] = min(grad[i], 0.1)

    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * (grad)
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad  * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region
    if proj:
        
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

    # if score == 0:
    #     lr = max(lr / 2, 0.001)
    # if score > best_score:
    #     lr = min(lr * 2, 0.1)
    return lr

def ADAM2(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj, beta, z, q=1):
    cur_loss = losses[0]
    for i in range(q):
        grad[i] = q*(losses[i+1] - losses[0])* z[i] / beta
    # argument indice should be removed for the next version
    # the entire modifier is updated for every epoch and thus indice is not required
    avg_grad = np.mean(grad, axis=0)
    #print('avg grad shape ', avg_grad.shape)
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * avg_grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
    vt_arr[indice] = vt

    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt)  + 1e-8 )
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

    return real_modifier, adam_epoch, mt_arr, vt_arr

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # def sign(x):
    #     return np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / (2*9.9999)
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (9.9999 * 9.9999)
    # print("New epoch:")
    # print('grad', grad)
    # hess[hess < 0] = 1.0
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # negative hessian cannot provide second order information, just do a gradient descent
    hess[hess < 0] = 1.0
    # hessian too small, could be numerical problems
    hess[hess < 0.1] = 0.1
    # print(hess)
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * grad / hess
    #  it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print('delta', old_val - m[indice])
    m[indice] = old_val
    # print(m[indice])


@jit(nopython=True)
def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / (9.9999 * 2)
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (9.9999 * 9.9999)
    # print("New epoch:")
    # print(grad)
    # print(hess)
    # positive hessian, using newton's method
    hess_indice = (hess >= 0)
    # print(hess_indice)
    # negative hessian, using ADAM
    adam_indice = (hess < 0)
    # print(adam_indice)
    # print(sum(hess_indice), sum(adam_indice))
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # print(adam_indice)
    # Newton's Method
    m = real_modifier.reshape(-1)
    old_val = m[indice[hess_indice]] 
    old_val -= lr * grad[hess_indice] / hess[hess_indice]
    # set it back to [-0.5, +0.5] region
 
    m[indice[hess_indice]] = old_val
    # ADMM
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch[adam_indice]))) / (1 - np.power(beta1, epoch[adam_indice]))
    old_val = m[indice[adam_indice]] 
    old_val -= lr * corr * mt[adam_indice] / (np.sqrt(vt[adam_indice]) + 1e-8)
    # old_val -= lr * grad[adam_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[adam_indice]]), down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1
    # print(m[indice])

class BlackBoxL2:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, print_every=100,
                 early_stop_iters=0,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 use_log=False, use_tanh=True, use_resize=False, adam_beta1=0.9, adam_beta2=0.999,
                 reset_adam_after_found=False,
                 solver="adam", attack="basic", save_ckpts="", load_checkpoint="", start_iter=0,
                 init_size=32, use_importance=True, method='tanh', dct=True, dist_metrics="", maximize="plus", htype="phash", height=288, width=288, channels=1, theight=288, twidth=288, tchannels=1, multi_imgs_num=1, mc_sample = 2):
        """
        The L_2 optimized attack. 
        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.
        Returns adversarial examples for the supplied model.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """
        image_height, image_width, num_channels = height, width, channels
        target_height, target_width, target_channels = theight, twidth, tchannels
        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.attack = attack
        self.LEARNING_RATE = learning_rate
        self.learning_rate = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_width = image_width
        self.image_height = image_height
        self.resize_init_size = init_size
        self.use_importance = use_importance
        self.feature_extractor = load_extractor('high_extract')

        self.multi_imgs_num = multi_imgs_num
        self.mc_sample = mc_sample

        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_height
            self.small_y = image_width

        self.use_tanh = use_tanh
        self.use_resize = use_resize
        self.save_ckpts = save_ckpts
        if save_ckpts:
            print('creating save_ckpts ', save_ckpts)
            os.system("mkdir -p {}".format(save_ckpts))

        self.repeat = binary_search_steps >= 10
        self.method = method
        self.dct = dct
        self.maximize = maximize
        # self.post_success_num_rand_vec = num_rand_vec
        # self.num_rand_vec = self.post_success_num_rand_vec
        self.dist_metrics = dist_metrics
        print('use tanh', use_tanh)
        print('method is ', self.method)
        print('batch size ', self.batch_size)
        print('resize ', self.resize_init_size)
        print('use important ', self.use_importance)
        print('load ckpt', load_checkpoint)
        print('use dct', dct) 
        print('dist metrics', self.dist_metrics) 
        print("htype ", htype)
        self.htype = htype

        # each batch has a different modifier value (see below) to evaluate
        # small_shape = (None,self.small_x,self.small_y,num_channels)
        shape = (None,image_height,image_width,num_channels)
        single_shape = (image_height, image_width, num_channels)  #grayscale  #self.image_shape
        target_shape = (target_height, target_width, target_channels)
        self.single_shape = single_shape
        self.single_shape2 = (image_height, image_width, num_channels * 3) # rgb
        small_single_shape = (self.small_x, self.small_y, num_channels)    # self.modifier_shape
        self.small_single_shape = small_single_shape
        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size
        if self.use_resize:
            self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
            # scaled up image
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_height, image_width])
            # operator used for resizing image
            self.resize_size_x = tf.placeholder(tf.int32)
            self.resize_size_y = tf.placeholder(tf.int32)
            self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
            self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y])
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, image_height, image_width, num_channels))
            # no resize
            self.scaled_modifier = self.modifier
        # the real variable, initialized to 0
        self.load_checkpoint = load_checkpoint
        if load_checkpoint:
            # if checkpoint is incorrect reshape will fail
            print("Using checkpint", load_checkpoint)
            self.real_modifier = np.load(load_checkpoint).reshape((1,) + small_single_shape)
        else:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
            # self.real_modifier = np.random.randn(self.image_height * self.image_width* self.num_channels).astype(np.float32).reshape((1,) + single_shape)
            # self.real_modifier /= np.linalg.norm(self.real_modifier) 
            #self.initialize_modifier()
        # these are variables to be more efficient in sending data to tf
        # we only work on 1 image at once; the batch is for evaluation loss at different modifiers
        self.timg = tf.Variable(np.zeros(single_shape), dtype=tf.float32)  # grayscale
        self.timg_multi = tf.Variable(np.zeros((self.multi_imgs_num, ) + single_shape), dtype=tf.float32)  # grayscale
        self.thimg = tf.Variable(np.zeros(target_shape), dtype=tf.float32)
        # self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        self.assign_timg = tf.placeholder(tf.float32, single_shape) 
        self.assign_timg_multi = tf.placeholder(tf.float32, (self.multi_imgs_num, ) + single_shape) 
        self.assign_thimg = tf.placeholder(tf.float32, target_shape) 
        # self.assign_tlab = tf.placeholder(tf.float32, num_labels)
        self.assign_const = tf.placeholder(tf.float32)
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # broadcast self.timg to every dimension of modifier
        if use_tanh:
            
            l = []

            for i in range(self.timg_multi.shape[0]):
                l.append(self.scaled_modifier[0] + self.timg_multi[i])
            
            for i in range(self.timg_multi.shape[0]):
                s = self.scaled_modifier + self.timg_multi[i]
                l = tf.concat([l, s[1:self.batch_size * self.mc_sample + 1]], axis=0)

            self.newimg = tf.tanh(l) / 2

        #     self.newimg = tf.tanh(tf.concat([(self.scaled_modifier + self.timg1)[0:1, :, :, :], (self.scaled_modifier + self.timg2)[0: 1, :, :, :], (self.scaled_modifier + self.timg1)[1:self.batch_size * 4 + 1, :, :, :], (self.scaled_modifier + self.timg2)[1:self.batch_size * 4 + 1, :, :, :]], axis=0)) / 2
        
        # self.newimg = tf.tanh(self.scaled_modifier + self.timg)/2
        # self.scaled_modifier = self.batch_size * self.mc_sample + 1

        # (self.scaled_modifier + self.timg1)[0:1, :, :, :]
        # (self.scaled_modifier + self.timg1)[2:self.batch_size + 2, :, :, :]
        
        # (self.scaled_modifier + self.timg2)[1:2, :, :, :]
        # (self.scaled_modifier + self.timg2)[self.batch_size + 2: self.batch_size * 2 + 2, :, :, :]


            # (65, 375, 500, 1)

            # self.newimg = tf.tanh(self.scaled_modifier + self.timg)/2
            #self.newimg2 = tf.py_function(self.colorize, [self.newimg], tf.float32)

            # prepare for perceptual metrics
            self.newimg_grgb =  tf.image.grayscale_to_rgb(self.newimg + 0.5)
            self.timg_grgb = tf.image.grayscale_to_rgb(tf.tanh(self.timg)/2+ 0.5)
            self.timg_grgb = tf.reshape(self.timg_grgb, (1,image_height,image_width,num_channels*3))

        else:
            self.newimg = self.scaled_modifier + self.timg    #grayscale
            # not finished for rgb
        
        # prediction BEFORE-SOFTMAX of the model
        # now we have output at #batch_size different modifiers
        # the output should have shape (batch_size, num_labels)


        # self.output = model.predict(self.newimg)
        # #
        # print(self.output)
        if self.htype == "phash":
       
            if use_tanh:
                #self.output2 = model.predict1(self.newimg, tf.tanh(self.timg) / 2, self.method)
                self.output2 = model.predict1(self.newimg, tf.tanh(self.thimg) / 2, self.method, self.TARGETED)
            
            else:
                self.output2 = model.predict1(self.newimg, self.thimg, self.method, self.TARGETED)
                #self.output2 = model.predict1(self.newimg2, self.timg2, self.method)

        elif self.htype == "pdqhash":
            if use_tanh:
                self.output2 = model.predict_pdq(self.newimg, tf.tanh(self.thimg), self.TARGETED)
            else:
                self.output2 = model.predict_pdq(self.newimg, self.thimg, self.TARGETED)
        
        else: # elif self.htype == "blockhash": 
            if use_tanh:
                self.output2 = model.predict2(self.newimg, tf.tanh(self.thimg) / 2, self.TARGETED)
            
            else:
                self.output2 = model.predict2(self.newimg, self.thimg, self.TARGETED)           
 
        
        self.default_graph = tf.get_default_graph()
        # producer_version = default_graph.graph_def_versions.producer

        self.cache_dir = os.path.expanduser('~/.lpips')
        os.makedirs(self.cache_dir, exist_ok=True)
            # files to try. try a specific producer version, but fallback to the version-less version (latest).
        pb_fnames = [
            #'%s_%s_v%s_%d.pb' % ('net-lin', 'alex', '0.1', producer_version),
            '%s_%s_v%s.pb' % ('net-lin', 'alex', '0.1'),
        ]
        for pb_fname in pb_fnames:
            if not os.path.isfile(os.path.join(self.cache_dir, pb_fname)):
                try:
                    _download(os.path.join(_URL, pb_fname), self.cache_dir)
                except urllib.error.HTTPError:
                    pass
            if os.path.isfile(os.path.join(self.cache_dir, pb_fname)):
                break

        with open(os.path.join(self.cache_dir, pb_fname), 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        if self.dist_metrics == "pdist": 
            # calculate perceptual metrics distances here
            self.input0 = self.newimg_grgb
            self.input1 = self.timg_grgb
            # self.input2 = self.timg2_grgb

            self.batch_shape = tf.shape(self.input0)[:-3]
            
            # adjust for multiple batches dimension automatically
            self.input0 = tf.reshape(self.input0, tf.concat([[-1], tf.shape(self.input0)[-3:]], axis=0))
            self.input1 = tf.reshape(self.input1, tf.concat([[-1], tf.shape(self.input1)[-3:]], axis=0))

            # NHWC to NCHW
            self.input0 = tf.transpose(self.input0, [0, 3, 1, 2])
            self.input1 = tf.transpose(self.input1, [0, 3, 1, 2])
            # self.input2 = tf.transpose(self.input2, [0, 3, 1, 2])
            # self.loss3= tf.reduce_sum((self.input0-self.input2), [1,2,3])
            input0_name, input1_name = '0:0', '1:0'

            self.input0 = self.input0 * 2.0 - 1.0
            self.input1 = self.input1 * 2.0 - 1.0
            # self.input2 = self.input2 * 2.0 - 1.0
            # print('input0 shape after 2========', self.input0)



            def calculate_dist(graph_def, input0_name, input0, input1_name, input1):
                _ = tf.import_graph_def(graph_def,
                                            input_map={input0_name: input0, input1_name: input1})
                distance, = self.default_graph.get_operations()[-1].outputs
                return distance

                # _ = tf.import_graph_def(graph_def,
                #                         input_map={input0_name: input0, input1_name: input1})
                # distance, = default_graph.get_operations()[-1].outputs
            distance1 = calculate_dist(self.graph_def, input0_name, self.input0, input1_name, self.input1)
            self.distance1 = distance1
                #distance2 = calculate_dist(graph_def, input0_name, self.input0, input1_name, self.input2)

            if distance1.shape.ndims == 4:
                distance1 = tf.squeeze(distance1, axis=[-3, -2, -1])
                self.final_distance = distance1
                # if distance2.shape.ndims == 4:
                #     distance2 = tf.squeeze(distance2, axis=[-3, -2, -1])


            # reshape the leading dimensions
            
            self.pdist = tf.reshape(distance1, self.batch_shape) # scaler
            # self.pdist2 = tf.reshape(distance2, batch_shape)
            # self.pdist = self.sess.run(self.pdist)
            self.l2dist = self.pdist
            # self.l2dist = 0.25 * self.pdist + 0.75 * self.pdist2
            # self.l2dist_trans = self.pdist2
            self.loss2 = self.l2dist
        else:

        # sum up the losses (output is a vector of #batch_size)
            # self.l2dist = 0.25 * self.l2dist_original + 0.75 * self.l2dist_trans
            # self.loss2 = self.l2dist
            if use_tanh:
                self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2), [1,2,3])
            else:
                self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.timg), [1,2,3])

            self.l2dist = self.l2dist
            self.loss2 = self.l2dist


        self.loss1 = self.output2 * self.const
        # print('what is ', self.CONFIDENCE)
#        self.real1 = tf.maximum(0.0, self.real1 - self.other1 +self.CONFIDENCE)
        # put robust distance here, aims at minimizing robust distance
       
        self.loss = self.loss1 + self.loss2 
     
        
        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.timg_multi.assign(self.assign_timg_multi))
        self.setup.append(self.thimg.assign(self.assign_thimg))
        self.setup.append(self.const.assign(self.assign_const))
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.used_var_list = np.zeros(var_size, dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        self.var_size = var_size

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype = np.float32)
        self.modifier_down = np.zeros(var_size, dtype = np.float32)

        # random permutation for coordinate update
        self.perm = np.random.permutation(var_size)
        self.perm_index = 0

        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        # self.beta1 = 0.8
        # self.beta2 = 0.99
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        self.stage = 0
        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype = np.float32)
        self.hess = np.zeros(batch_size, dtype = np.float32)
        # for testing
        self.grad_op = tf.gradients(self.loss, self.modifier)
        # self.grad2 = np.zeros((self.num_rand_vec, var_size), dtype = np.float32)
        # self.hess2 = np.zeros((self.num_rand_vec, var_size), dtype = np.float32)
        # compile numba function
        # self.coordinate_ADAM_numba = jit(coordinate_ADAM, nopython = True)
        # self.coordinate_ADAM_numba.recompile()
        # print(self.coordinate_ADAM_numba.inspect_llvm())
        # np.set_printoptions(threshold=np.nan)
        # set solver
        solver = solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            print("unknown solver", solver)
            self.solver = coordinate_ADAM
        print("Using", solver, "solver")

    def gen_image_robust(self, timg, newimg):
        if self.num_channels == 1:
            # timg = np.repeat(timg, self.small_single_shape2, axis=0) # to rgb version
            # newimg = np.repeat(newimg, self.small_single_shape2, axis=0)
            timg = tf.image.grayscale_to_rgb((timg+0.5) * 255)
            newimg = tf.image.grayscale_to_rgb((newimg+0.5)*255)
 
        else:
            timg = (timg+0.5) * 255
            newimg = (newimg+0.5) * 255
    
        return timg, newimg

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i+size, j:j+size] = np.max(image[i:i+size, j:j+size])
        
        return img_pool

    def avg_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i+size, j:j+size] = np.average(image[i:i+size, j:j+size])
        
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double = False):
        if self.num_channels == 3: 
            prev_modifier = np.squeeze(prev_modifier)
            old_shape = prev_modifier.shape
        else: 
            prev_modifier = prev_modifier[0]
            old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0]*2, old_shape[1]*2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype = np.float32)
        for i in range(prev_modifier.shape[2]):
            if self.dct:
                pixels = prev_modifier[:,:,i] 
                dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
                image = np.abs(dct)
                #image_pool = self.avg_pooling(image, old_shape[0] // 8)
            else:
                image = np.abs(prev_modifier[:,:,i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            #image_pool = self.avg_pooling(image, old_shape[0] // 8)
            if gen_double:
                # prob[:,:,i] = scipy.misc.imresize(image_pool, 2.0, 'nearest', mode = 'F')
                # prob[:,:,i] = resize(image_pool, (width * 2, height * 2), 'nearest')
                prob[:,:,i] = cv2.resize(image_pool, (image_pool.shape[0]*2, image_pool.shape[1] * 2))
                prob[:,:,i] = resize(image_pool, (image_pool.shape[0]*2, image_pool.shape[1] * 2), anti_aliasing=True)
            else:
                prob[:,:,i] = image_pool
        prob /= np.sum(prob)
        return prob


    def resize_img(self, small_x, small_y, reset_only = False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
            #self.real_modifier = np.random.randn(self.image_height * self.image_width* self.num_channels).astype(np.float32).reshape((1,) + self.single_shape)
            #self.real_modifier /= np.linalg.norm(self.real_modifier) 
            #self.initialize_modifier()
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(self.real_modifier)
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x, self.resize_size_y: self.small_y, self.resize_input: self.real_modifier})
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

    def fake_blackbox_optimizer(self):
        true_grads, losses, l2s, loss1, loss2, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
        # ADAM update
        grad = true_grads[0].reshape(-1)
        
        epoch = self.adam_epoch[0]
        mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grad)
        corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)
        # print(grad.shape, mt.shape, vt.shape, self.real_modifier.shape)
        # m is a *view* of self.real_modifier
        m = self.real_modifier.reshape(-1)
        # this is in-place
        m -= self.LEARNING_RATE * corr * (mt / (np.sqrt(vt) + 1e-8))
        self.mt = mt
        self.vt = vt
        # m -= self.LEARNING_RATE * grad
        if not self.use_tanh:
            m_proj = np.maximum(np.minimum(m, self.modifier_up), self.modifier_down)
            np.copyto(m, m_proj)
        self.adam_epoch[0] = epoch + 1
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]


    def blackbox_optimizer(self, iteration, bestscore):
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.batch_size * self.mc_sample + 1, axis=0)
        var_size = self.real_modifier.size
        if self.use_importance:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False, p = self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False)
        indice = self.var_list[var_indice]
        # indice = self.var_list
        # regenerate the permutations if we run out
        # if self.perm_index + self.batch_size >= var_size:
        #     self.perm = np.random.permutation(var_size)
        #     self.perm_index = 0
        # indice = self.perm[self.perm_index:self.perm_index + self.batch_size]
        # b[0] has the original mo difier, b[1] has one index added 0.0001
        for i in range(self.batch_size):
            for j in range(self.mc_sample // 2):
                var[i * self.mc_sample + j + 1].reshape(-1)[indice[i]] += (self.mc_sample // 2 - j) * delta
                var[i * self.mc_sample + self.mc_sample - j].reshape(-1)[indice[i]] -= (self.mc_sample // 2- j) * delta
            # var[i * 4 + 1].reshape(-1)[indice[i]] += 2 * delta 
            # var[i * 4 + 2].reshape(-1)[indice[i]] += delta 
            # var[i * 4 + 3].reshape(-1)[indice[i]] -= delta 
            # var[i * 4 + 4].reshape(-1)[indice[i]] -= 2 * delta 
        
        #losses, l2s, loss1, loss2, loss3, scores, nimgs = self.sess.run([self.loss, self.l2dist, self.loss1, self.loss2, self.loss3, self.output2, self.newimg], feed_dict={self.modifier: var})
        losses, l2s, loss1, loss2, scores, nimgs= self.sess.run([self.loss, self.l2dist, self.loss1, self.loss2, self.output2, self.newimg], feed_dict={self.modifier: var})
        self.LEARNING_RATE = self.solver(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh, self.multi_imgs_num, self.mc_sample)
        # print("verifying============= ", self.adam_epoch)
        # adjust sample probability, sample around the points with large gradient
        # if self.save_ckpts:
        #     np.save('{}/iter{}'.format(self.save_ckpts, iteration), self.real_modifier)

        # if self.real_modifier[0].shape[0] > self.resize_init_size:
        #     # print('real modifier shape ', self.real_modifier.shape)
        #     self.sample_prob = self.get_new_prob(self.real_modifier)
        #     self.sample_prob = self.sample_prob.reshape(var_size)

        if self.use_resize:
            # print('real modifier shape!!! ', self.real_modifier.shape)
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)
        # if the gradient is too small, do not optimize on this variable
        # self.var_list = np.delete(
        # self.var_list, indice[np.abs(self.grad) < 5e-3])
        # reset the list every 10000 iterations
        # if iteration%200 == 0:
        #    print("{} variables remained at last stage".format(self.var_list.size))
        #    var_size = self.real_modifier.size
        #    self.var_list = np.array(range(0, var_size))

        # print('loss3 is for verification======= ', loss3[0], loss3[1], loss3[2])
        print(loss1)

        loss_out = loss1[0:self.multi_imgs_num].sum()
        scores_out = scores[0:self.multi_imgs_num].sum()
        return losses[0], l2s[0], loss_out, loss2[0], scores_out, nimgs[0], nimgs[1], nimgs, loss1[0:self.multi_imgs_num]

    def initialize_modifier(self):
        self.real_modifier = np.zeros((1,) + self.small_single_shape, dtype=np.float32)
        var_size = self.real_modifier.size
        self.beta = 1/(var_size) * 1000

        var_noise = np.random.normal(loc=0, scale=3000, size=(1, var_size)) 
        #print("var noise", var_noise)
        noise_norm = np.apply_along_axis(np.linalg.norm, 1, var_noise, keepdims=True)
        var_noise = var_noise/noise_norm
        self.real_modifier += self.beta*var_noise.reshape(self.num_rand_vec, self.small_x, self.small_y, self.num_channels)
        #var = np.concatenate((self.real_modifier, self.real_modifier + self.beta*var_noise.reshape(self.num_rand_vec, self.small_x, self.small_y, self.num_channels)), axis=0)
    
    def attack(self, imgs):
        """
        Perform the L_2 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        # we can only run 1 image at a time, minibatches are used for gradient evaluation
        for i in range(0,len(imgs)):

            r.extend(self.attack_batch(imgs[i]))
        return np.array(r)

    # only accepts 1 image at a time. Batch is used for gradient evaluations at different points
    def attack_batch(self, gray_img, multi_gray_imgs, img, targetHashimg, img_id):
   
    # def attack_batch(self, gray_img, gray_img1, gray_img2, img, targetHashimg, img_id):
        # remove the extra batch dimension
        if len(img.shape) == 4:
            img = img[0]
        if len(gray_img.shape) == 4:
            gray_img = gray_img[0]
        if len(targetHashimg.shape) == 4:
            targetHashimg = targetHashimg[0]

        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img*1.999999)
            gray_img = np.arctanh(gray_img*1.999999)  # literally times 2
            multi_gray_imgs = np.arctanh(multi_gray_imgs*1.999999)
            targetHashimg = np.arctanh(targetHashimg*1.999999)
            

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10
       
        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)
        gray_img = gray_img.astype(np.float32)
        multi_gray_imgs = multi_gray_imgs.astype(np.float32)
        targetHashimg = targetHashimg.astype(np.float32)
    
        # set the upper and lower bounds for the modifier
        if not self.use_tanh:

            self.modifier_up = 0.5 - gray_img.reshape(-1)
            self.modifier_down = -0.5 - gray_img.reshape(-1)
        self.LEARNING_RATE = self.learning_rate  # initial learning rate
        
        if not self.load_checkpoint:
            if self.use_resize: # resize the modifier, scale up for the real modifier
                self.resize_img(self.resize_init_size, self.resize_init_size, True)
            else:
                self.real_modifier.fill(0.0)
        # the best l2, score, and image attack

        o_bestl = 1e10

        o_best_const = CONST
        o_bestl2 = 1e10
        
        loss_x = []
        loss_y = []

        o_bestscore= 1
        o_bestattack = gray_img
        o_bestattack_second = gray_img 
        L3 = False
        o_bestscore2 = 1
        o_bestl22 = 1e10
        o_bestattack2 = gray_img
        count = 0  # for iterations
        o_iterations_first_attack = 0
        modifier = self.real_modifier
        print('new image L3', L3)
        binary_search_step = self.BINARY_SEARCH_STEPS
        for outer_step in range(binary_search_step):

            lx = []
            ly_sum = []
            ly = []

            bestl2 = 1e10
            bestscore = 1
            L3_count = 0
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound
            
            self.sess.run(self.setup, {self.assign_timg: gray_img, self.assign_timg_multi: multi_gray_imgs, self.assign_thimg: targetHashimg, self.assign_const: CONST})
            prev = 1e10
            train_timer = 0.0
            last_loss1 = 1.0
            
            if not self.load_checkpoint:
                if self.use_resize:
                    self.resize_img(self.resize_init_size, self.resize_init_size, True)
                else:
                    self.real_modifier.fill(0.0)
            # reset ADAM status
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
            # self.LEARNING_RATE = self.learning_rate 
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            if self.solver_name != "fake_zero":
                multiplier = 24
            for iteration in range(self.start_iter, self.MAX_ITERATIONS):
                # print(self.MAX_ITERATIONS) = 50000
#                print(self.start_iter, iteration, self.MAX_ITERATIONS)
                count += 1
                # if self.use_resize:
                #     if iteration == 2000:
                #     # if iteration == 2000 // 24:
                #         self.resize_img(64,64)
                #     # if iteration == 2000 // 24 + (10000 - 2000) // 96:
                #     if iteration == 3000: 
                #         self.resize_img(128,128)
                #     if iteration == 5000:
                #         self.resize_img(256,256)
                    
                loss, loss1, loss2 = self.sess.run((self.loss,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier})
                
                # print("shape")
                # print(loss.shape, loss1.shape, loss2.shape)
                if iteration%(self.print_every) == 0:
                    print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, lr={}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, loss[0], loss1[0], loss2[0], self.LEARNING_RATE))
                    sys.stdout.flush()
                
                #     sys.stdout.flush()
                attack_begin_time = time.time()
                # perform the attack 
                if self.solver_name == "fake_zero":
                    l, l2, loss1, loss2, score, nimg = self.fake_blackbox_optimizer()
                elif self.solver_name == "adam" or self.solver_name == "newton" or self.solver_name == "adam_newton":
                    # l, l2, loss1, loss2, loss3, score, nimg = self.blackbox_optimizer(iteration)
                    l, l2, loss1, loss2, score, nimg, second_nimg, nimgs, loss1es = self.blackbox_optimizer(iteration, bestscore)

                # l = self.blackbox_optimizer(iteration)
                if self.solver_name == "fake_zero":
                    eval_costs += np.prod(self.real_modifier.shape)
                elif self.solver_name =="adam" or self.solver_name =="newton" or self.solver_name == "adam_newton":
                    eval_costs += self.batch_size

                lx.append(iteration)
                ly_sum.append(loss1)
                ly.append(loss1es)
                prev_modifier_saved = np.copy(self.real_modifier)


                if self.save_ckpts: 
                    if score < o_bestscore2:
                        o_bestscore2 = score
                        o_bestl22 = l2
                        print('saving outstep {} and iteration {}'.format(outer_step, iteration))
                        np.save('{}/best_modifier_img{}'.format(self.save_ckpts, img_id), prev_modifier_saved)
                        o_bestattack2 = nimg
                        modifier = prev_modifier_saved

                    elif score == o_bestscore2:
                        if l2 < o_bestl22:
                            o_bestl22 = l2
                            print('saving outstep {} and iteration {}'.format(outer_step, iteration))
                            np.save('{}/best_modifier_img{}'.format(self.save_ckpts, img_id), prev_modifier_saved)
                            o_bestattack2 = nimg
                            modifier = prev_modifier_saved

                    
                # reset ADAM states when a valid example has been found
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    # we have reached the fine tunning point
                    # reset ADAM to avoid overshoot
                    if self.reset_adam_after_found:
                        self.mt.fill(0.0)
                        self.vt.fill(0.0)
                        self.adam_epoch.fill(1)
                    self.stage = 1
                last_loss1 = loss1

                if l2 < bestl2 and score <= 0:
                    bestl2 = l2
#                    bestscore = np.argmax(score)
                    bestscore = score
#               

                if l < o_bestl:
                    o_bestl = l
                    o_bestattack2 = nimg

                if l2 < o_bestl2  and score <= 0: # and score != 1:

                    if o_bestl2 == 1e10:
                        #print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, loss3 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2, loss3))
                        print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2))
                        sys.stdout.flush()
                        L3_count +=1
                        L3 = True
                        o_iterations_first_attack = count
                    else:
                        L3_count += 1
                        print("[STATS][L3](Attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2))
                        sys.stdout.flush()
                    o_bestl2 = l2
                    o_bestscore = score
                    o_bestattack = nimgs
                    modifier = prev_modifier_saved  
                    o_bestattack_second = second_nimg
                    o_best_const = CONST
                    #print('bestscore ', o_bestscore)
                elif score <= 0:
                    print("do I get here")
                    L3_count +=1
                    print("[STATS][L3](Attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2))
                    sys.stdout.flush()
                else:
                    if o_bestscore != 0:
                        o_iterations_first_attack = count
                train_timer += time.time() - attack_begin_time

                if self.ABORT_EARLY:
                    if L3_count == 1 and score<=0:
                        print("Early stopping because we found the adversarial example")
                        break

                if (self.ABORT_EARLY and iteration % self.early_stop_iters == 0):
                    if l > prev*.9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = l

            loss_x.append(lx)
            loss_y.append(ly_sum)
            loss_x.append(lx)
            loss_y.append(ly)

            # adjust the constant as needed
            if bestscore <= 0:
                # success, divide const by two
                print('old constant: ', CONST)
                upper_bound = min(upper_bound,CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound)/2
                print('new constant: ', CONST)
    

            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                print('old constant: ', CONST)
                lower_bound = max(lower_bound,CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound)/2
                else:
                    CONST *= 10
                print('new constant: ', CONST)
           
        # return the best solution found

        # h1 = imagehash.phash(gen_image(o_bestattack)) 
        # print(h1)

        return o_bestattack, o_bestattack_second, o_best_const, L3, o_bestattack2, o_iterations_first_attack, nimgs, modifier, loss_x, loss_y
