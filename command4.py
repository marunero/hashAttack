import os
import time




# 06_21_11_45_scaled_modifier.npy
# os.system('python main.py -i 13 -n 1 -mu 2 --gpu 0 -mc 64 --targeted -hash phash256 -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.01 -mi 20000 -pc 64 --save result --seed 12352')

# 06_21_17_38_scaled_modifier.npy
# os.system('python main.py -i 13 -n 1 -mu 6 --gpu 0 -mc 22 --targeted -hash phash256 -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.01 -mi 20000 -pc 64 --save result --seed 12392')


os.system('python main.py -i 13 -n 1 -mu 8 -ti 13 --gpu 0 -mc 16 --targeted -hash phash256 -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.01 -mi 20000 -pc 64 --save result --seed 12692')
