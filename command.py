import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 15 -n 4 -mu 1 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.5 -mi 16000')