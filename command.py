import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 17 -n 1 -mu 1 -mc 32 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 500 -mi 2000')