import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 13 -n 1 -mu 3 --gpu 0 -mc 64 --targeted -hash phash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.01 --use_resize --resize_size 64 -mi 20000 -pc 64 --save result --seed 25146')
