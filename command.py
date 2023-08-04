import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


# --use_resize --resize_size 64 

# lr = 0.25
# perturbation_const = 1

# for i in range(2):
#     perturbation_const = 1
#     for j in range(7):
#         os.system('python main.py -i 17 -n 2 -mu 1 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr ' + str(lr) + ' -mi 20000 -pc ' + str(perturbation_const))

#         perturbation_const *= 2
#     lr *= 2

# lr 0.25 -> perturbation_cost 8~16
# lr 0.5 -> perturbation_cost 16~32

# --use_resize --resize_size 64

# 6/5/20/00
# os.system('python main.py -i 20 -n 1 -mu 1 --gpu 0 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.05 -mi 20000 -pc 100 --save result --seed 1840')

# 6/5/18/31
# os.system('python main.py -i 15 -n 5 -mu 1 --gpu 0 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.05 --use_resize --resize_size 64 -mi 20000 -pc 16 --save result/photoDNA --seed 12345')



# # photoDNA, use_resize, pc = 8 ~ 16, lr = 0.01
# os.system('python main.py -i 18 -n 1 -mu 1 --gpu 0 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 --use_resize --resize_size 64 -lr 0.01 -mi 20000 -pc 16 --save result --seed 2547')

# # PDQ, use_resize, pc = 100, lr = 0.1
# os.system('python main.py -i 18 -n 1 -mu 1 --gpu 0 -mc 64 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.5 -mi 20000 -pc 16 --save result --seed 2548')

# # pdq_photoDNA, use_resize, pc = ?, lr = ?
# os.system('python main.py -i 18 -n 1 -mu 1 --gpu 0 -mc 64 --targeted -hash pdq_photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 --use_resize --resize_size 64 -lr 0.01 -mi 20000 -pc 16 --save result --seed 2549')



# os.system('python main.py -i 19 -n 1 -mu 1 --gpu 0 -mc 32 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 64 --batch 1 -lr 0.001 -mi 40000 -pc 16 --save result --seed 2546 --checkpoint result/06_12_20_47.npy')

# os.system('python main.py -i 10 -n 1 -mu 7 --gpu 0 -mc 32 --targeted -hash pdqhash -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 20000 -pc 100 --save result --seed 25146')

# os.system('python main.py -i 18 -n 1 -mu 3 --gpu 0 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.02 -mi 20000 -pc 32 --save result --seed 25146')


# category2 -> target2
# PDQ, n=1, 2, 4, 8, 16 
# selection strategy - minimum

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 32 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.1 -mi 2000 -pc 1 --save result')

os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 16 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.1 -mi 2000 -pc 1 --save result')

os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 16 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.1 -mi 4000 -pc 1 --save result')

os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 16 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.1 -mi 8000 -pc 1 --save result')



os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 16 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 2000 -pc 1 --save result')
os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 16 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr 0.1 -mi 4000 -pc 1 --save result')


# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 8 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.0005 -mi 4000 -pc 1 --save result')

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 8 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.0005 -mi 8000 -pc 1 --save result')

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 8 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.0005 -mi 16000 -pc 1 --save result')

# os.system('python main.py -i 0 -n 1 -mu 1 -ti 0 --gpu 0 -mc 8 --targeted -hash SIFT -dist l2dist --optimizer momentum --use_grayscale --use_resize --resize_size 128 --batch 1 -lr 0.001 -mi 16000 -pc 1 --save result')