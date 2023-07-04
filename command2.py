import os
import time

# category2 -> target2
# PDQ, n=1, 2, 4, 8, 16 
# selection strategy - minimum

os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 128 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 2 -ti 0 --gpu 0 -mc 64 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 4 -ti 0 --gpu 0 -mc 32 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 6 -ti 0 --gpu 0 -mc 20 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 0 -n 1 -mu 8 -ti 0 --gpu 0 -mc 16 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

os.system('python main.py -i 0 -n 1 -mu 12 -ti 0 --gpu 0 -mc 10 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

os.system('python main.py -i 0 -n 1 -mu 16 -ti 0 --gpu 0 -mc 8 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

os.system('python main.py -i 0 -n 1 -mu 20 -ti 0 --gpu 0 -mc 6 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')