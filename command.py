import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 0 -n 1 -mu 1 -mc 4 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.005 -mi 200')