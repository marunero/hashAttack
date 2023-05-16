import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))

for i in range(1, 2):
    lr = 0.5

    os.system('python main.py -i 15 -n 4 -mu 1 -mc 32 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr ' + str(lr * i) + ' -mi 8000')