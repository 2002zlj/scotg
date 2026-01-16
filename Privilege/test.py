import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置相对路径的起始目录
# os.chdir('/data_25T/zlj/FSCIL/PriViLege-main_me')
import argparse
import importlib
from utils.utils import *
import torch
from models.base.ViT_fscil_trainer import ViT_FSCILTrainer,CLIP_ViT_FSCILTrainer

MODEL_DIR=None
DATA_DIR = './local_datasets/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-save_root', type=str, default='/.result/')
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-out', type=str, default=None)

    # about pre-training
    # parser.add_argument('-epochs_base', type=int, default=200)
    parser.add_argument('-epochs_base', type=int, default=50)
    # parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=20)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=2e-4)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=80)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_comb', 'ft_euc']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='2, 3')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=-1)
    # parser.add_argument('-seeds', nargs='+', help='<Required> Set flag', required=True, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-rotation', action='store_true')
    parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    parser.add_argument('-vit', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-clip', action='store_true')
    
    parser.add_argument('-ED', action='store_true')
    parser.add_argument('-ED_hp', type=float, default=0.1)
    
    parser.add_argument('-LT', action='store_true')
    parser.add_argument('-WC', action='store_true')
    parser.add_argument('-MP', action='store_true')
    
    
    parser.add_argument('-SKD', action='store_true')
    
    parser.add_argument('-l2p', action='store_true')
    parser.add_argument('-dp', action='store_true')
    
    parser.add_argument('-prefix', action='store_true')
    parser.add_argument('-pret_clip', action='store_true')
    parser.add_argument('-comp_out', type=int, default=1.)
    parser.add_argument('-clip_text', type=str)
    
    # parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-taskblock', type=int, default=2)
    parser.add_argument('-ft', action='store_true')
    parser.add_argument('-lp', action='store_true')
    parser.add_argument('-PKT_tune_way', type=int, default=1)
    parser.add_argument('-shot', type=str, default='5shot')
    
    return parser


if __name__ == '__main__':
    torch.set_num_threads(2)
    devices = [d for d in range(torch.cuda.device_count())]
    device_names  = [torch.cuda.get_device_name(d) for d in devices]
    print('GPU COUNT',torch.cuda.device_count())
    print("Devices:",devices)
    print("Device Name:",device_names)
    
    parser = get_command_line_parser()
    args = parser.parse_args()
    ###########################################################
    args.save_root = '/data_25T/zlj/FSCIL/PriViLege-main_me/result'
    args.project = 'base'
    args.dataset = 'classroom'
    args.base_mode = 'ft_dot'
    args.new_mode = 'avg_cos'
    args.gamma = 0.1
    args.lr_base = 2e-4
    args.lr_new = 2e-4
    args.decay = 0.0005
    args.epochs_new = 3
    args.schedule = 'Cosine'
    args.gpu ='0'
    args.temperature = 16
    args.start_session = 0
    args.batch_size_base = 128
    args.vit = True
    args.comp_out = 1
    args.ED = True
    args.SKD = True
    args.LT = True
    args.out ='PriViLege'
    # args.dataroot = '/data_25T/zlj/FSCIL/PriViLege-main_me/local_datasets/khpark'
    milestones_list = ['20 30 45']
    ###########################################################
    seeds = [2022]
    epochs_bases = [5] #5
    shots = ['1shot','3shot','5shot']
    for seed in seeds:
        for shot in shots:
            print("Pretraining -- Seed{}".format(seed))
            for i, epochs_base in enumerate(epochs_bases):
                args.seed = seed
                args.epochs_base = epochs_base
                args.milestones = milestones_list[i]
                args.shot = shot
                if shot!='5shot':
                    args.dataroot ='./dataset_reqs/FSCIL/classroom_'+shot+'/b2/'
                    args.dataset = 'classroom_'+shot
                else:
                    args.dataroot ='./dataset_reqs/FSCIL/classroom/b2/'
                    args.dataset = 'classroom'    
            
                set_seed(args.seed)
                pprint(vars(args))
                args.num_gpu = set_gpu(args)
                if args.vit:
                    if args.baseline:
                        trainer = importlib.import_module('models.%s.ViT_fscil_Baseline_trainer' % (args.project)).ViT_Baseline_FSCILTrainer(args)
                    #* L2P / DP
                    elif args.l2p:
                        pass
                    elif args.dp:
                        pass
                    else:
                        trainer = ViT_FSCILTrainer (args)
                    
                elif args.pret_clip:
                    trainer = CLIP_ViT_FSCILTrainer(args)
                else:
                    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
                trainer.test()