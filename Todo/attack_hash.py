import sys
import os
import tensorflow.compat.v1 as tf
import numpy as np
import math
import cv2

from numba import jit
from PIL import Image


class BlackBoxL2:
    def __init__(self, sess, model, batch_size=1,
                 targeted=True, learning_rate=0.1,
                 binary_search_steps=1, max_iterations=4000,
                 initial_loss_const=1, use_resize=False, resize_size=32, use_grayscale=False, 
                 adam_beta1=0.99, adam_beta2=0.9999, mc_sample = 2, 
                 solver="adam", hash_metric="phash", save_ckpts="", load_checkpoint="",
                 dist_metric="l2dist", input_x=288, input_y=288, input_c=1, target_x=288, target_y=288, target_c=3, multi_imgs_num=1):
        self.sess = sess
        self.model = model
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.mc_sample = mc_sample
        self.multi_imgs_num = multi_imgs_num

        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_loss_const = initial_loss_const
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        
        self.hash_metric = hash_metric
        self.dist_metric = dist_metric

        self.use_resize = use_resize

        self.resized_x = input_x
        self.resized_y = input_y
        if self.use_resize:
            self.resized_x = resize_size
            self.resized_y = resize_size

        input_shape = (input_x, input_y, input_c)
        target_shape = (target_x, target_y, target_c)
        self.input_shape = input_shape

        self.resized_shape = (self.resized_x, self.resized_y, input_c)

        if self.use_resize:
            self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
            # scaled up image
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_height, image_width])
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, input_x, input_y, input_c))
            # no resize
            self.scaled_modifier = self.modifier

        self.real_modifier = np.zeros((1,) + self.resized_shape, dtype=np.float32)

        self.input_images = tf.Variable(np.zeros((self.multi_imgs_num, ) + input_shape), dtype=tf.float32) 
        self.target_image = tf.Variable(np.zeros(target_shape), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        self.assign_input_images = tf.placeholder(tf.float32, (self.multi_imgs_num, ) + input_shape) 
        self.assign_target_image = tf.placeholder(tf.float32, target_shape) 
        self.assign_const = tf.placeholder(tf.float32)



        if use_tanh:
            
            l = []

            for i in range(self.timg_multi.shape[0]):
                l.append(self.scaled_modifier[0] + self.timg_multi[i])
            
            for i in range(self.timg_multi.shape[0]):
                s = self.scaled_modifier + self.timg_multi[i]
                l = tf.concat([l, s[1:self.batch_size * self.mc_sample + 1]], axis=0)

            self.newimg = tf.tanh(l) / 2
            
            if self.use_grayscale:
                self.newimg_grgb =  tf.image.grayscale_to_rgb(self.newimg + 0.5)
                self.timg_grgb = tf.image.grayscale_to_rgb(tf.tanh(self.timg)/2+ 0.5)
                self.timg_grgb = tf.reshape(self.timg_grgb, (1,image_height,image_width,num_channels*3))
            else:
                self.newimg_grgb = self.newimg
                self.timg_grgb = self.timg

        else:
            self.newimg = self.scaled_modifier + self.timg
        

        
        if self.hash_metric == "phash":
       
            if use_tanh:
                
                self.output2 = model.predict1(self.newimg, tf.tanh(self.thimg) / 2, self.method, self.TARGETED)
            
            else:
                self.output2 = model.predict1(self.newimg, self.thimg, self.method, self.TARGETED)
                

        elif self.hash_metric == "pdqhash":
            if use_tanh:
                self.output2 = model.predict_pdq(self.newimg, tf.tanh(self.thimg), self.TARGETED)
            else:
                self.output2 = model.predict_pdq(self.newimg, self.thimg, self.TARGETED)
        
        elif self.hash_metric == "photoDNA":
            if use_tanh:
                self.output2 = model.predict_photoDNA(self.newimg, tf.tanh(self.thimg), self.TARGETED)
            else:
                self.output2 = model.predict_photoDNA(self.newimg, self.thimg, self.TARGETED)

        else: # elif self.hash_metric == "blockhash": 
            if use_tanh:
                self.output2 = model.predict_photoDNA(self.newimg, tf.tanh(self.thimg), self.TARGETED)
            else:
                self.output2 = model.predict_photoDNA(self.newimg, self.thimg, self.TARGETED)
 
        if self.dist_metrics == "l2dist": 
            if use_tanh:
                self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2), [1,2,3])
            else:
                self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.timg), [1,2,3])
            self.loss2 = self.l2dist


        self.loss1 = self.output2 * self.const
       
        self.loss = self.loss1 + self.loss2 
     
        
        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.input_images.assign(self.assign_input_images))
        self.setup.append(self.target_image.assign(self.assign_target_image))
        self.setup.append(self.const.assign(self.assign_const))
        
        # prepare the list of all valid variables
        var_size = self.resized_x * self.resized_y * input_c
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
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        self.stage = 0

        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype = np.float32)
        self.hess = np.zeros(batch_size, dtype = np.float32)
        
        # if solver == "adam":
        self.solver = coordinate_ADAM
        
        print("Using", solver, "solver")


    def blackbox_optimizer(self, iteration, bestscore):
        return

    def attack_batch(self, input_images, target_image):
        lower_bound = 0.0
        CONST = self.initial_loss_const
        upper_bound = 1e10

        for i in range(self.binary_search_step):
            self.sess.run(self.setup, {self.assign_input_images: input_images, self.assign_target_image: target_image, self.assign_const: CONST})

            # reset ADAM status
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)

            for iteration in range(self.max_iterations):
                loss, loss1, loss2 = self.sess.run((self.loss,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier})
                
                
                if iteration%(self.print_every) == 0:
                    print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, lr={}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, loss[0], loss1[0], loss2[0], self.LEARNING_RATE))
                    sys.stdout.flush()

        return