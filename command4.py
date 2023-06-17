import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 13 -n 1 -mu 3 --gpu 0 -mc 64 --targeted -hash phash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 40000 -pc 64 --save result --seed 3452')
