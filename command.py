import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 0 -n 1 -mu 1 -mc 8 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 1 -mi 2000')
