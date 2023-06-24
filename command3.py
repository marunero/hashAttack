import os
import time



# os.system('python main.py -i 18 -n 1 -mu 2 --gpu 0 -mc 32 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.005 -mi 20000 -pc 16 --save result --seed 232')


# 17 & 18 & 19 -> 18 
os.system('python main.py -i 17 -n 1 -mu 3 -ti 18 --gpu 0 -mc 42 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.01 -mi 20000 -pc 16 --save result --seed 30172')