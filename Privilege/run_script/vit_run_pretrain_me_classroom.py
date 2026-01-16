import os
import sys
# seeds = [1,2,3,4,5]
seeds = [2022]

project = 'base'
# dataset = 'mini_imagenet'
dataset = 'classroom'
# dataset = 'cub200'

lr_base = 0.00325
lr_new = 0.00325

epochs_bases = [5] #5
epochs_new = 5 #3
milestones_list = ['20 30 45']

# #* data_dir = '/local_datasets/'
# data_dir = sys.argv[1]
# gpu_num = sys.argv[2]
data_dir = '/data_25T/zlj/FSCIL/PriViLege-main_me/local_datasets/khpark'
gpu_num = 0

for seed in seeds:
    print("Pretraining -- Seed{}".format(seed))
    for i, epochs_base in enumerate(epochs_bases):
        os.system(''
                'python /data_25T/zlj/FSCIL/PriViLege-main_me/train.py '
                '-project {} '
                '-dataset {} '
                '-base_mode ft_dot '
                '-new_mode avg_cos '
                '-gamma 0.1 '
                '-lr_base {} '
                '-lr_new {} '
                '-decay 0.0005 '
                '-epochs_base {} '
                '-epochs_new {} '
                '-schedule Cosine '
                '-milestones {} '
                '-gpu {} '
                '-temperature 16 '
                '-start_session 0 '
                '-batch_size_base 32 '
                '-seed {} '
                '-vit '
                # '-clip'
                '-comp_out 1 '
                # '-prefix '
                '-ED '
                '-SKD '
                '-LT '
                '-out {} '
                '-dataroot {}'.format(project, dataset, lr_base, lr_new, epochs_base, epochs_new, milestones_list[i], gpu_num, seed, 'PriViLege', data_dir)
                )
