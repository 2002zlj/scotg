import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import clip



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Classroom_dataset(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, is_clip=False):
        temp_model, train_transforms, val_transforms = clip.load("ViT-B/16", device=device, jit=False)
        del temp_model        
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = root
        # self.root = '/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/FSCIL/classroom/b2/'
        self.file_path = self.root+'classroom_'+setname+'.txt'

        with open('/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/classroom_classes.txt', "r") as f:
            lines = f.read().splitlines()
        self.class_names =  [line.split("\t")[-1] for line in lines]
        self.class_names = np.array(self.class_names)

        self.all_data, self.all_targets = [], []

        with open(self.file_path, "r") as f:
            for line in f:
                split_line = line.split(" ")
                path = split_line[0].strip()
                # x.append(os.path.join(self.data_path, path))
                self.all_data.append(path)
                self.all_targets.append(int(split_line[-1].strip()))
        self.all_data, self.all_targets = np.array(self.all_data), np.array(self.all_targets)

        self.data = []
        self.targets = []
        for i in range(len(self.all_targets)):
            if self.all_targets[i] in index:
                self.data.append(self.all_data[i])
                self.targets.append(self.all_targets[i])
        self.data, self.targets = np.array(self.data), np.array(self.targets)

        self.classes = {}
        for i in range(len(self.data)):
            self.classes[self.data[i]] = self.class_names[self.targets[i]]

        

        if train:
            self.transform = transforms.Compose(train_transforms)
            # if base_sess:
            #     self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            # else:
            #     self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose(val_transforms)
            # self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def mapping_clsidx_to_txt(self):
        classes = {}
        lines = [x.strip() for x in open('/data/pgh2874/FSCIL/Ours/dataloader/miniimagenet/map_clsloc.txt', 'r').readlines()][1:]
        for l in lines:
            name, class_num, class_txt = l.split(' ')
            if name not in classes.keys():
                classes[name]=class_txt
        return classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform.transforms(Image.open(path).convert('RGB'))
        return image, targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_2.txt"
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '../../data'

    trainset = Classroom_dataset(root=dataroot, train=True, transform=None, index_path=txt_path)
    # print(trainset.targets)
    cls = np.unique(trainset.targets)
    batch_size_base = len(trainset)
    print(batch_size_base)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
