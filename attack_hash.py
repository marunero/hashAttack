import sys
import os
import tensorflow.compat.v1 as tf
import numpy as np
import math
import cv2
import time 

from numba import jit
from PIL import Image

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, multi_imgs_num, mc_sample, delta):
    for i in range(batch_size):
        grad[i] = 0

        for j in range(multi_imgs_num):
            for k in range(mc_sample // 2):
                grad[i] += losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + k]
                grad[i] -= losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + (mc_sample // 2) + k]
        grad[i] /= delta * (mc_sample // 2) * (1 + mc_sample // 2)

    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * (grad)
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad  * grad)
    vt_arr[indice] = vt

    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)

    old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print(grad)
    # print(old_val - m[indice])

    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

    return lr

@jit(nopython=True)
def momentum(losses, real_modifier, lr, grad, perturbation, batch_size, multi_imgs_num, mc_sample, perturbation_pixel, resized_shape, up, down):
    g = np.zeros((resized_shape), dtype=np.float32)
    # for i in range(batch_size):
    #     for j in range(multi_imgs_num):
    #         for k in range(mc_sample // 2):
    #             c_k = max(losses[j] - losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + k], 0)
    #             c_k -= max(losses[j] - losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + (mc_sample - 1) - k], 0)
    #             g += c_k * perturbation[k]

    for i in range(batch_size):
        for j in range(multi_imgs_num):
            for k in range(mc_sample // 2):
                a = losses[j] - losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + k]
                b = losses[j] - losses[multi_imgs_num + (j * mc_sample * batch_size) + (i * mc_sample) + (mc_sample - 1) - k]

                if a > 0 and b > 0:
                    if a > b:
                        g += (a) * perturbation[k] * perturbation_pixel
                    else:
                        g += (-b) * perturbation[k] * perturbation_pixel
                else:
                    g += (a - b) * perturbation[k] * perturbation_pixel
                
                # g += (a - b) * perturbation[k] * perturbation_pixel


    g /= multi_imgs_num * mc_sample

    # add momentum
    grad = 0.4999 * grad + 0.5001 * g 
    # grad = g

    # normalization
    if np.sum(grad ** 2) != 0:
        grad = grad / np.sum(grad ** 2)

    real_modifier += grad * lr * perturbation_pixel
    # real_modifier = np.clip(real_modifier, down, up)

    return lr

class hash_attack:
    def __init__(self, sess, model, batch_size=1,
                 targeted=True, learning_rate=0.1,
                 binary_search_steps=1, max_iterations=4000, print_unit=1,
                 initial_loss_const=1, use_tanh=True, use_resize=False, resize_size=32, use_grayscale=False, 
                 adam_beta1=0.99, adam_beta2=0.9999, mc_sample = 2, multi_imgs_num=1, perturbation_const=10,
                 solver="adam", hash_metric="phash",
                 dist_metric="l2dist", input_x=288, input_y=288, input_c=1, target_x=288, target_y=288, target_c=3):
        self.sess = sess
        self.model = model
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.mc_sample = mc_sample
        self.multi_imgs_num = multi_imgs_num
        self.batch_size = batch_size


        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_loss_const = initial_loss_const
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.print_unit = print_unit

        self.hash_metric = hash_metric
        self.dist_metric = dist_metric

        # PhotoDNA threshold
        if self.hash_metric == "photoDNA":
            self.threshold = 1800
        elif self.hash_metric == "pdqhash":
            self.threshold = 90
        else: # phash, etc.
            self.threshold = 0

        self.use_resize = use_resize
        self.use_tanh = use_tanh

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
            self.scaled_modifier = tf.image.resize_images(self.modifier, [input_x, input_y])
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, input_x, input_y, input_c))
            # no resize
            self.scaled_modifier = self.modifier
        
        # self.scaled_modifier = tf.tanh(self.scaled_modifier)

        self.real_modifier = np.zeros((1, ) + self.resized_shape, dtype=np.float32)

        self.input_images = tf.Variable(np.zeros((self.multi_imgs_num, ) + input_shape), dtype=tf.float32) 
        self.target_image = tf.Variable(np.zeros(target_shape), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        self.assign_input_images = tf.placeholder(tf.float32, (self.multi_imgs_num, ) + input_shape) 
        self.assign_target_image = tf.placeholder(tf.float32, target_shape) 
        self.assign_const = tf.placeholder(tf.float32)


        l = []

        for i in range(self.input_images.shape[0]):
            l.append(self.scaled_modifier[0] + self.input_images[i])

        for i in range(self.input_images.shape[0]):
            s = self.scaled_modifier + self.input_images[i]
            l = tf.concat([l, s[1:self.batch_size * self.mc_sample + 1]], axis=0)
        self.newimg = l
        self.output1 = model.get_loss(self.newimg, self.target_image, self.hash_metric)
        
        self.loss1 = self.output1

        # if self.dist_metrics == "l2dist": 
        l2 = []
        for i in range(self.input_images.shape[0]):
            l2.append(self.scaled_modifier[0])
        for i in range(self.input_images.shape[0]):
            l2 = tf.concat([l2, self.scaled_modifier[1:self.batch_size * self.mc_sample + 1]], axis=0)

        if self.use_tanh == True:
            self.l2dist = tf.reduce_sum(tf.square(tf.tanh(l2) / 2), [1, 2, 3])
        else:
            self.l2dist = tf.reduce_sum(tf.square(l2), [1, 2, 3])
        self.loss2 = self.l2dist


        self.loss = self.loss1 + self.loss2 * self.const


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

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype = np.float32)
        self.modifier_down = np.zeros(var_size, dtype = np.float32)

        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype = np.float32)
        self.hess = np.zeros(batch_size, dtype = np.float32)
        self.up = np.zeros(self.resized_shape, dtype = np.float32)
        self.down = np.zeros(self.resized_shape, dtype = np.float32)


        self.solver_metric = solver
        if self.solver_metric == "adam":
            self.solver = coordinate_ADAM
        elif self.solver_metric == "momentum":
            self.solver = momentum
        self.momentum = 0.5
        self.perturbation_pixel = 0.004
        self.perturbation_const = perturbation_const
        self.p = np.array([np.random.normal(loc = 0, scale = self.perturbation_pixel, size = self.resized_shape) for j in range(self.mc_sample // 2)])

        self.delta = 0.49999 / (mc_sample / 2)

        print("Using", solver, "solver")


    def blackbox_optimizer(self):
        var = np.repeat(self.real_modifier, self.batch_size * self.mc_sample + 1, axis=0)


        # per pixel

        if self.solver_metric == 'adam':
            var_size = self.real_modifier.size
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False)

            # print("var size = ", var_size)
            # print("var_indice = ", var_indice)


            indice = self.var_list[var_indice]

            # print("indice = ", indice)

            for i in range(self.batch_size):
                for j in range(self.mc_sample // 2):
                    var[i * self.mc_sample + j + 1].reshape(-1)[indice[i]] += (self.mc_sample // 2 - j) * self.delta
                    var[i * self.mc_sample + self.mc_sample - j].reshape(-1)[indice[i]] -= (self.mc_sample // 2- j) * self.delta


            losses, loss1, loss2 = self.sess.run([self.loss, self.loss1, self.loss2], feed_dict={self.modifier: var})

            print(loss1)

            lr = self.solver(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up, self.modifier_down, self.learning_rate, self.adam_epoch, self.beta1, self.beta2, self.multi_imgs_num, self.mc_sample, self.delta)

        # elif self.solver_metric == 'momentum':
        else:
            self.p = np.array([np.random.normal(loc = 0, scale = self.perturbation_pixel, size = self.resized_shape) for j in range(self.mc_sample // 2)])

            # self.p = np.clip(self.p, self.down, self.up)

            for i in range(self.batch_size):
                for j in range(self.mc_sample // 2):
                    var[i * self.mc_sample + j + 1] += self.p[j] * self.perturbation_const
                    var[i * self.mc_sample + self.mc_sample - j] -= self.p[j] * self.perturbation_const

            losses, loss1, loss2, scaled_modifier, nimgs = self.sess.run([self.loss, self.loss1, self.loss2, self.scaled_modifier, self.newimg], feed_dict={self.modifier: var})
 
            lr = self.solver(losses, self.real_modifier, self.learning_rate, self.grad, self.p, self.batch_size, self.multi_imgs_num, self.mc_sample, self.perturbation_pixel, self.resized_shape, self.up, self.down)      
            print(losses)     
            # print(loss2)




        return losses[0:self.multi_imgs_num], loss1[0:self.multi_imgs_num], loss2[0:self.multi_imgs_num], nimgs[0:self.multi_imgs_num], scaled_modifier[0]

    def attack_batch(self, input_images, target_image):
        lower_bound = 0.0
        CONST = self.initial_loss_const
        upper_bound = 1e10

        self.modifier_up = 0.5 - input_images[0].reshape(-1)
        self.modifier_down = -0.5 - input_images[0].reshape(-1)


        # if self.use_resize == True:
        #     self.up = 1 - 
        #     self.down = 
        # else:
        #     self.up = 1 - np.max(input_images, axis=0)
        #     self.down = 0 - np.min(input_images, axis=0)
        
        best_loss_x = []
        best_loss_y = []
        success = False

        for binary_step in range(self.binary_search_steps):
            self.sess.run(self.setup, {self.assign_input_images: input_images, self.assign_target_image: target_image, self.assign_const: CONST})

            train_timer = 0.0


            self.real_modifier.fill(0.0)

            # reset ADAM status
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
        
            self.grad.fill(0.0)

            loss_x = []
            loss_y = []

            start_time = time.time()

            for iteration in range(self.max_iterations):
                loss, loss1, loss2 = self.sess.run((self.loss,self.loss1,self.loss2), feed_dict={self.modifier: self.real_modifier})

                if iteration % (self.print_unit) == 0:
                    hours = int(train_timer // 3600)
                    minutes = int(train_timer // 60)
                    seconds = int(train_timer % 60)
                    print("[STATS] iter = {}, time = {}:{}:{}, modifier shape = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, hours, minutes, seconds, self.real_modifier.shape, loss[0:self.multi_imgs_num].sum(), loss1[0:self.multi_imgs_num].sum(), loss2[0:self.multi_imgs_num].sum()))
                    sys.stdout.flush()
                
                loss_x.append(iteration)
                loss_y.append(loss)

                l, hashdiffer, loss2, modified_imgs, scaled_modifier = self.blackbox_optimizer()

                if hashdiffer.max() <= self.threshold:
                    success = True

                    break

                end_time = time.time()

                train_timer = end_time - start_time

                # l = self.blackbox_optimizer()

                if success == True:
                    continue
                else:
                    continue 
                    

        return success, self.modifier[0], loss_x, loss_y, hashdiffer, modified_imgs, scaled_modifier