import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


os.system('python main.py -i 0 -n 1 -mu 2 -mc 2 --targeted -hash photoDNA -dist l2dist --optimizer adam --use_grayscale --use_resize --resize_size 64 --batch 8')
