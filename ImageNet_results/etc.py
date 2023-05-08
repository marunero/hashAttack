import os
import time
path = os.getcwd()
os.system('cd {}'.format(os.getcwd()))


timestamp = []
pre_t = time.time()

dataset = 12
start = 1
dataset = dataset - (start - 1)


n_data = 1
batch = 64



def done(index, file_list):
    for i in range(len(file_list)):
        if str(index) in file_list[i]:
            return True
    return False

h = [32]

attack = ['transform_crop', 'transform_scale', 'transform_rot']
save = ['Tcrop', 'Tscale', 'Trot']



# li = [[80, 0.01]]
li = [[20000, 0.02]]
# cmd = 'python ../test_attack_black.py --translateRGB --attack basic -d imagenet --maxiter 2000 --reset_adam -n 1 --solver adam -b 1 -p 1 --hash 10 --use_resize --method "tanh" --batch 64 --gpu 0 --lr 0.005 -s target --start_idx=0 --dist_metrics "pdist"'

for i in range(len(li)):
    iteration = str(li[i][0])
    lr = str(li[i][1])

    # untargeted 

    # os.system('python ../test_attack_black.py --untargeted -c 1 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 6 -n 1 --solver adam -b 1 -p 1 --hash 10 --use_resize --init_size 64 --method "tanh" --batch 16 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "pdist"')

    # targeted

    # --use_resize --init_size 64 
    # --load_ckpt np/best_modifier_img0.npy 
    
    os.system('python ../test_attack_black.py -ht pdqhash -c 10 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 2 -n 1 -mu 1 -mc 4 --batch 16 --lr ' + lr + ' --solver adam -b 1 -p 1 --hash 32 --method "tanh" --use_resize --init_size 128 --gpu 0 -s target  --dist_metrics "l2dist" --save_ckpts np --seed ' + str(1600 + i))
