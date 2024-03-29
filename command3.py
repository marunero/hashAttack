import os
import time



# category2 -> target2
# PDQ, n=1, 2, 4, 8, 16 
# selection strategy - sort as hash values far away in series (start from target image)

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 128 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --use_resize --resize_size 64 --gpu 0 -mc 128 --targeted -hash ahash256 -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.05 -mi 2000 -pc 100 --save result --seed 25146 --check result/07_04_17_46_modifier.npy')

os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 128 --targeted -hash phash64 -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.1 -mi 10000 -pc 64 --save result --seed 25146')


# os.system('python main.py -i 0 -n 1 -mu 4 -ti 0 --gpu 0 -mc 32 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 6 -ti 0 --gpu 0 -mc 20 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 8 -ti 0 --gpu 0 -mc 16 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 12 -ti 0 --gpu 0 -mc 10 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 16 -ti 0 --gpu 0 -mc 8 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 20 -ti 0 --gpu 0 -mc 6 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')