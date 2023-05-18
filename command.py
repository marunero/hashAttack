import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


# --use_resize --resize_size 64 

lr = 0.04
for i in range(6):
    
    os.system('python main.py -i 17 -n 2 -mu 1 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr ' + str(lr) + ' -mi 20000')

    lr *= 2

