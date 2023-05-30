import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


# --use_resize --resize_size 64 

# lr = 0.25
# perturbation_const = 1

# for i in range(2):
#     perturbation_const = 1
#     for j in range(7):
#         os.system('python main.py -i 17 -n 2 -mu 1 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr ' + str(lr) + ' -mi 20000 -pc ' + str(perturbation_const))

#         perturbation_const *= 2
#     lr *= 2

# lr 0.25 -> perturbation_cost 8~16
# lr 0.5 -> perturbation_cost 16~32




os.system('python main.py -i 17 -n 2 -mu 2 --gpu 0 -mc 64 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.01 -mi 20 -pc 16 --save result')