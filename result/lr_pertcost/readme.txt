
lr = 0.25
perturbation_const = 1

for i in range(2):
    perturbation_const = 1
    for j in range(7):
        os.system('python main.py -i 17 -n 2 -mu 1 -mc 64 --targeted -hash photoDNA -dist l2dist --optimizer momentum --use_grayscale --batch 1 -lr ' + str(lr) + ' -mi 20000 -pc ' + str(perturbation_const))

        perturbation_const *= 2
    lr *= 2

