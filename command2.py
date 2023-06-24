import os
import time

# 06_21_11_28_scaled_modifier.npy
# os.system('python main.py -i 17 -n 1 -mu 1 -ti 18 --gpu 0 -mc 128 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.01 -mi 20000 -pc 16 --save result --seed 231')

# 18 -> 18 modifier continually
# os.system('python main.py -i 17 -n 1 -mu 1 -ti 18 --gpu 0 -mc 128 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.01 -mi 20000 -pc 16 --save result --seed 3002 --checkpoint result/06_21_11_28_modifier.npy')



# 17 & 18 -> 18 
# os.system('python main.py -i 19 -n 1 -mu 1 -ti 19 --gpu 0 -mc 128 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.02 -mi 30000 -pc 16 --save result --seed 3142')



os.system('python main.py -i 0 -n 1 -mu 6 -ti 0 --gpu 0 -mc 20 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

os.system('python main.py -i 0 -n 1 -mu 12 -ti 0 --gpu 0 -mc 10 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

os.system('python main.py -i 0 -n 1 -mu 20 -ti 0 --gpu 0 -mc 6 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')
