
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import json
import hydra
import logging
from omegaconf import DictConfig
import sys
from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.datasets import build_cl_scenarios
import numpy as np
import random
import shutil
import clip
import argparse
from transformers import CLIPModel
import time
from models.model import FSCIL_classrroom_SCOTG_expand,FSCIL_classrroom_SCOTG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Train CPE-CLIP model on a few-shot class incremental learning task.")
parser.add_argument("--L_g", type=int, default=2, help="NuNumber of prompts to be used in the encoders")
parser.add_argument("--deep_g", type=int, default=12, help="Number of layers of the encoders in which the prompts will be processed")
parser.add_argument("--text_deep_replace_method", type=str, default="replace", choices=["replace", "accumalate", "accumulate_same"], help="Method to replace the text prompts in the encoders. Options: replace, accumulate, accumulate_same")
parser.add_argument("--vision_deep_replace_method", type=str, default="accumulate", choices=["replace", "accumulate", "accumulate_same"], help="Method to replace the vision prompts in the encoders. Options: replace, accumulate, accumulate_same")
parser.add_argument("--dataset_name", type=str, default="cifar100", choices=["cifar100", "cub200", "miniimagenet"], help="Name of the dataset to be used. Options: cifar100, cub200, miniimagenet")
parser.add_argument("--n_runs", type=int, default=5, help="Number of runs to be executed", required=False)
parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to be used in the runs", required=False)
parser.add_argument("--ablation", type=str, choices=["no_accumulation", "no_regularization", "no_vision_prompts"])
parser.add_argument("--model_path", type=str)
parser.add_argument("--regularization_method", type=str)
parser.add_argument("--manual_prompt", type=str)
parser.add_argument("--use_scheduler")
parser.add_argument("--eval_mb_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--origin", type=int)

args = parser.parse_args()

args.L_g=2
args.deep_g=12
args.dataset_name='classrooma_visual_llava_CLIP'
args.n_runs=5


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



config_path = os.path.dirname(__file__)+'/'+ 'config'
config_name = 'classroom_32_20-3-ZSCL.yaml'
@hydra.main(config_path=config_path, config_name=config_name, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    cfg.class_order =os.path.dirname(__file__)+'/'+"class_orders/classroom_b2.yaml"
    cfg.train_file_path = os.path.dirname(__file__)+'/'+ 'dataset_reqs/FSCIL/classroom/'+os.path.basename(cfg.class_order ).split('.')[0].split('_')[-1]
    cfg.scenario ='fscil'
    cfg.method = 'Ablation_'+abla
    cfg.L_g=2
    cfg.deep_g=12
    cfg.text_deep_replace_method='replace'
    cfg.vision_deep_replace_method='accumulate'
    cfg.regularization_method = 'balance'
    cfg.visual=1
    cfg.n_runs=5    
    cfg.model_name = os.path.dirname(__file__)+'/'+  'ckp/clip-vit-base-patch16'
    cfg.workdir = os.path.dirname(os.path.abspath(__file__))

    cfg.a1 = 0.1
    cfg.a2 = 0.1
    cfg.a3 = 0.1

    cfg.first = 5
    cfg.second = 8
    cfg.third = 10

    current_timestamp = time.time()
    local_struct = time.localtime(current_timestamp)  # 本地时区
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", local_struct)
    result_save_path = os.path.dirname(__file__)+'/'+  'result/'+cfg.method+'/'+time_str
        
    utils.save_config(cfg)
    cfg.class_order = utils.get_class_order(cfg.class_order)
    
    if cfg.method == 'Ablation_only_expand':
        model  = FSCIL_classrroom_SCOTG_expand(cfg, device)
    else:
        cfg.a1 = 0.1
        cfg.a2 = 0
        cfg.a3 = 0
        cfg.first = 11
        cfg.second = 11
        cfg.third = 11
        model  = FSCIL_classrroom_SCOTG(cfg, device)
    zero_model = CLIPModel.from_pretrained(cfg.model_name)
    zero_model.cuda()
    zero_model.eval()
    temp_model, train_transforms, val_transforms = clip.load("ViT-B/16", device=device, jit=False)
    del temp_model

    test_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, mode='test', transforms=val_transforms
    )

    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, mode='train',transforms=train_transforms
    )

    model.classes_names = classes_names

    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list_test = []
    metric_logger_test = Logger(list_subsets=["test"])
    os.makedirs(result_save_path,exist_ok=True)
    os.makedirs(result_save_path+'/log',exist_ok=True)
    os.makedirs(result_save_path+'/ckp',exist_ok=True)
    cfg.log_path = result_save_path+'/log/'+cfg.method+'.log'

    main_save_path = result_save_path+'/log/main.txt'
    shutil.copy(os.path.join(config_path,config_name),result_save_path+'/log/'+config_name)
    shutil.copy(os.path.abspath(__file__),main_save_path)

    ckp_root = cfg.workdir+'/'+ "weights/"+ cfg.method +"/ckp"

    for task_id, _ in enumerate(test_dataset):
        logging.info(f"Evaluation for task {task_id} has started.")
        model.test(task_id, cfg, train_dataset, train_classes_names)
        model.model.eval()
        ckp_path = os.path.join(ckp_root,f'{task_id}.pth')
        ckp = torch.load(ckp_path)
        model.model.load_state_dict(ckp)
        model.model.cuda()

        test_loader = DataLoader(test_dataset[:task_id + 1], batch_size=cfg.batch_size)
        for inputs,targets,task_ids,names   in tqdm(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits_per_image = model(inputs)

            outputs = logits_per_image.softmax(dim=-1)
            metric_logger_test.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

        acc_list_test.append(100 * metric_logger_test.accuracy)
        with open(result_save_path+'/log/metrics_test.json', 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger_test.accuracy, 2),
            }) + '\n')
            metric_logger_test.end_task()


    with open(result_save_path+'/log/metrics_test.json', 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list_test[-1], 2), 
            'avg': round(statistics.mean(acc_list_test), 2)
        }) + '\n')

    del model
    del zero_model

if __name__ == "__main__":
    for abla in ['only_expand','only_refine']:
        seed_all(2022)
        continual_clip()   