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

# attack = ['basic', 'transform_ensemble', 'transform_crop', 'transform_scale', 'transform_rot']
# save = ['basic', 'ensemble', 'Tcrop', 'Tscale', 'Trot']


# lr = 0.005

# flag = True
# attackGoing = True

# while attackGoing:
#     attackGoing = False
#     for d in range(0, dataset):
#         d = d + (start - 1)
#         for i in range(len(h)):
#             for j in range(len(attack)):
#                 flag = True
                
#                 file_list = sorted(os.listdir("./" + save[j] + str(h[i])))
                
#                 for f in range(len(file_list)):
#                     if str(d + 1) == file_list[f][0:len(str(d + 1))]:
#                         flag = False
#                         break
                
#                 if flag == True:
#                     attackGoing = True
#                     cmd = 'python ../test_attack_black.py --translateRGB --untargeted --attack ' + attack[j] + ' --firstimg ' + str(d) + ' -d imagenet --reset_adam -n ' + str(n_data) + ' --solver adam -b 2 -p 1 --hash ' + str(h[i]) + ' --use_resize --method "tanh" --batch ' + str(batch) + ' --gpu 0 --lr ' + str(lr) +' -s ' + save[j] + str(h[i]) + ' --start_idx=0 --dist_metrics "pdist"' 
                    
#                     os.system(cmd)
#                     timestamp.append(time.time() - pre_t)
#                     pre_t = time.time()


# print(timestamp)

# li = [[4000 , 0.01]]
# # cmd = 'python ../test_attack_black.py --translateRGB --attack basic -d imagenet --maxiter 2000 --reset_adam -n 1 --solver adam -b 1 -p 1 --hash 10 --use_resize --method "tanh" --batch 64 --gpu 0 --lr 0.005 -s target --start_idx=0 --dist_metrics "pdist"'

# for i in range(len(li)):
#     iteration = str(li[i][0])
#     lr = str(li[i][1])

#     os.system('python ../test_attack_black.py -c 100 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 2 -n 1 --solver adam -b 2 -p 1 --hash 32 --use_resize --init_size 128 --method "tanh" --batch 32 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "pdist"')

# # 8 / 4000, 0.01
# # 30 / 4000, 


# li = [[80, 0.01]]
li = [[16000, 0.01]]
# cmd = 'python ../test_attack_black.py --translateRGB --attack basic -d imagenet --maxiter 2000 --reset_adam -n 1 --solver adam -b 1 -p 1 --hash 10 --use_resize --method "tanh" --batch 64 --gpu 0 --lr 0.005 -s target --start_idx=0 --dist_metrics "pdist"'

for i in range(len(li)):
    iteration = str(li[i][0])
    lr = str(li[i][1])

    # untargeted 

    # os.system('python ../test_attack_black.py --untargeted -c 1 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 6 -n 1 --solver adam -b 1 -p 1 --hash 10 --use_resize --init_size 64 --method "tanh" --batch 16 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "pdist"')

    # targeted

    os.system('python ../test_attack_black.py -ht pdqhash -c 10 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 0 -n 1 --solver adam -b 2 -p 1 --hash 32 --use_resize --init_size 64 --method "tanh" --batch 16 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "l2dist" --save_ckpts np --seed ' + str(1600 + i))


# for i in range(len(li)):
#     iteration = str(li[i][0])
#     lr = str(li[i][1])

#     # targeted

#     os.system('python ../test_attack_black.py -ht pdqhash -c 10000 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 1 -n 1 --solver adam -b 2 -p 1 --hash 32 --use_resize --init_size 128 --method "tanh" --batch 16 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "l2dist" --seed ' + str(1500 + i))


# for i in range(len(li)):
#     iteration = str(li[i][0])
#     lr = str(li[i][1])

#     # targeted

#     os.system('python ../test_attack_black.py -ht pdqhash -c 10000 --translateRGB --attack basic -d imagenet --maxiter ' + iteration + ' --reset_adam --start_idx 1 -n 1 --solver adam -b 2 -p 1 --hash 32 --use_resize --init_size 128 --method "tanh" --batch 8 --gpu 0 --lr ' + lr + ' -s target  --dist_metrics "l2dist" --seed ' + str(1600 + i))

# 8 / 4000, 0.01
# 30 / 4000, 