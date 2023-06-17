import os
import time




os.system('python main.py -i 18 -n 1 -mu 3 --gpu 0 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.005 -mi 20000 -pc 32 --save result --seed 25567')