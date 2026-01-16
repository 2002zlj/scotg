#from fcntl import F_SEAL_GROW
from math import fabs
from re import X
from torch.utils.data import Dataset, DataLoader
import torch
import random
import time
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
random.seed(2022)
#列别名称
classes = [
    "Listening",
    "reading",
    "Using_phone",
    "Using_pad",
    "Using_computers",
    "Scratching_head",
    "writing",
    "talking",
    "standing",
    "Sleeping",
    "teaching",
    "yawning",
    "walking",
    "relaxing",
    "analyzing",
    "Taking_bottle",
    "Gathering_up_bag",
    "drinking",
    "Taking_photos",
    "Listening_to_music",
    "discussing",
    "Setting_equipment",
    "Taking_bag",
    "Blackboard_writing",
    "Blackboard_wiping",
    "Taking_off",
    "Student_demonstrating",
    "eating",
    "reviewing",
    "Hands_up",
    "speaking",
    "Picking_up_computers"
]



classes_ZLJ = [
    "listening",
    "reading",
    "using_phone",
    "using_pad",
    "using_computers",
    "scratching_head",
    "writing",
    "talking",
    "standing",
    "sleeping",
    "teaching",
    "yawning",
    "walking",
    "relaxing",
    "analyzing",
    "taking_bottle",
    "gathering_up_bag",
    "drinking",
    "taking_photos",
    "listening_to_music",
    "discussing",
    "setting_equipment",
    "taking_bag",
    "blackboard_writing",
    "blackboard_wiping",
    "taking_off",
    "student_demonstrating",
    "eating",
    "reviewing",
    "hands_up",
    "speaking",
    "picking_up_computers"
]



#不同 小样本 增量类型
fscil_class = [
    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [23, 25, 21, 22],
        [30, 26, 28, 24],
        [17, 18, 29, 31],
        [20, 27, 16, 19],   ], # a1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [27, 25, 22],
        [28, 29, 26],
        [31, 23, 24],
        [20, 21, 30],      ], # a2

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],  ], # b1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22],
        [23, 24, 25],
        [26, 27, 28],
        [29, 30, 31],      ], # b2
]

#不同 常规连续学习 增量类型
cil_class = [
    [   [0, 1, 2, 3, 4, 5, 6, 7],
        [11, 19, 12, 16],
        [10, 13, 22, 21],
        [24, 30, 18, 14],
        [25, 26, 15, 17],
        [28, 31, 29, 27],
        [9,  20, 23,  8],    ], # a1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [31, 17, 26, 16],
        [21, 25, 27, 30],
        [18, 24, 29, 22],
        [13, 23, 14, 28],
        [19, 20, 12, 15],    ], # a2

    [   [0, 1, 2, 3, 4, 5, 6, 7],
        [8,  9,  10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],     ], # b1

    [   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],    ], # b2
]


class ClassRoom(Dataset):
    """
    一个用于加载不同类型小样本和连续学习数据集的类。
    """
    
    def __init__(self, fscil = True, train = True, classroom_type = 'resnet_a1', session = 0, f_shot = 5):
        
        # 根据参数选择预处理模型和增量类型
        self.pre_model, cil_type = classroom_type.strip().rsplit('_', 1)           
        self.curr_class = []
        self.kown_class = 0
        self.num_max = 500 #设置小样本训练阶段样本数量，和连续学习的样本数量
        self.classes = classes
        self.noise_level = 0.1
        
        if train==True:
            self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.RandomResizedCrop(size=(56, 56), scale=(0.8, 1.0)),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None),
                        ])
        else:
            self.transform=None
        
        
        
    
        #输入检查
        if self.pre_model not in ['resnet', 'vit', 'clipvit']:
            raise TypeError("预训练模型类型选择错误，可选[resnet、vit、clipvit]")
        if cil_type not in ['a1', 'a2', 'b1', 'b2']:
            raise TypeError("增量模式选择错误，可选[a1、a2、b1、b2]")

        #根据是否小样本，设置加载数据上限
        if fscil == True:
            if session > 0:
                self.num_max = f_shot
            class_list = fscil_class
        else:
            class_list = cil_class
        
        #根据选择预处理模型类型，设置不同的样本根目录
        root_base = '/media/dataset/dataset_36453_raw/image_features'
        self.root = os.path.join(root_base, self.pre_model)
            

        #根据不同的加载数据类设置，计算当前seesion的旧类数量
        #根据是否 train ，设置 txt 的文件目录
        #self.txt_path = '/amax/2020/gkl/CIL/make_data/classroom/index'   
        self.txt_path = '/amax/2020/qyl/PriViLege/data/index_list/classroom/clean/index'
        index = ['a1', 'a2', 'b1', 'b2'].index(cil_type)
        self.curr_class = class_list[index][session]
        if train:
            self.end = 'train/train_'
            if session > 0:
                if fscil == False and cil_type in ['a1', 'b1']:
                    self.kown_class = 8 + (session-1)*len(self.curr_class)
                if fscil == False and cil_type in ['a2', 'b2']:
                    self.kown_class = 12 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a1', 'b1']:
                    self.kown_class = 16 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a2', 'b2']:
                    self.kown_class = 20 + (session-1)*len(self.curr_class)
        else:
            if session != 0:
                temp = []
                for i in range(0, session):
                    temp += class_list[index][i]
                self.curr_class = temp + self.curr_class   
            self.end = 'test/test_'    
            self.num_max = 100
                  
                  
        # 当前增量阶段，单独保存每个类的数据路径和标签
        self.class_data_path = [[] for i in range(len(self.curr_class))]
        self.class_label = [[] for i in range(len(self.curr_class))]
        self.get_every_class_data()       
         
        # 一个增量阶段的总的数据路径和标签
        self.data_path = []
        self.targets = []
        self.get_all_class_data()


    def get_every_class_data(self,):
        """
        分别读取当前阶段每个类的数据和标签
        """
        for class_index in self.curr_class:
            #存储每个类的 txt 文件路径
            class_path = os.path.join(self.txt_path , (self.end + f'{class_index}.txt'))
            with open(class_path, 'r') as file:
                lines = file.readlines()
            unordered_index = list(range(len(lines)))# 将读取的数据打乱，防止顺序加载    
            random.shuffle(unordered_index)
            
            index = self.curr_class.index(class_index)            
            for i in unordered_index:
                data_path, label = lines[i].strip().rsplit(' ', 1) # 去掉行末的换行符并分割
                data_path = os.path.join(self.root, data_path)
                label = int(label)
                if(self.pre_model == 'clipvit'):
                    data_path = data_path.replace('.pkl', '_img.pkl')# 修改路径后缀
                if len(self.class_data_path[index]) < self.num_max:#当前数据量小于设定值, 继续向各类别列表中添加
                    self.class_data_path[index].append(data_path)
                    self.class_label[index].append(index + self.kown_class)
                else:
                    break
            
              
    def get_all_class_data(self, ):
        """
        整合当前增量阶段所有类的数据和标签
        """
        for class_index in range(len(self.curr_class)):
            for data in zip( self.class_data_path[class_index], self.class_label[class_index] ):
                data_path, label = data
                self.data_path.append(data_path)  
                self.targets.append(label)
                
    def add_noise(self,features, noise_std=0.1):
        noise = torch.randn_like(features) * noise_std
        return features + noise
                
    
    
    
        
    def random_crop_and_resize(self,embedding_map, crop_size=[12,12], target_size=[14,14]):
        """
        对二维嵌入网格随机裁剪并调整回原始大小。
        :param embedding_map: 嵌入网格 (H', W', D)，例如 (14, 14, 768)
        :param crop_size: 裁剪大小 (crop_h, crop_w)
        :param target_size: 调整后的目标大小 (target_h, target_w)
        :return: 调整尺寸后的嵌入网格
        """
        H, W, D = embedding_map.shape
        crop_h, crop_w = crop_size
        target_h, target_w = target_size

        if crop_h > H or crop_w > W:
            raise ValueError("裁剪尺寸不能超过嵌入网格的大小")

        # 随机选择裁剪的起始点
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # 裁剪操作
        cropped_map = embedding_map[top:top + crop_h, left:left + crop_w, :]

        # 调整回原始大小
        # torch.nn.functional.interpolate 要求输入为 4D 张量，因此需要添加批量维度
        cropped_map = cropped_map.permute(2, 0, 1).unsqueeze(0)  # (D, crop_h, crop_w) -> (1, D, crop_h, crop_w)
        resized_map = F.interpolate(cropped_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
        resized_map = resized_map.squeeze(0).permute(1, 2, 0)  # 恢复到 (target_h, target_w, D)

        return resized_map

    def random_flip(self,embedding_map, horizontal=True, vertical=False):
        """
        对二维嵌入网格随机翻转。
        :param embedding_map: 嵌入网格 (H', W', D)，例如 (14, 14, 768)
        :param horizontal: 是否随机水平翻转
        :param vertical: 是否随机垂直翻转
        :return: 翻转后的嵌入网格
        """
        if horizontal and random.random() > 0.5:
            embedding_map = torch.flip(embedding_map, dims=[1])  # 水平翻转
        if vertical and random.random() > 0.5:
            embedding_map = torch.flip(embedding_map, dims=[0])  # 垂直翻转
        return embedding_map

                
    def __len__(self):
        return len(self.data_path)


    def __getitem__(self, i):
        path, targets = self.data_path[i], self.targets[i]
        with open(path, 'rb') as file:
            data = torch.squeeze(torch.load(file)).detach().cpu()
            
        # data = data.view(14, 14, 768)
        # data = self.random_crop_and_resize(data )
        # data = self.random_flip(data )
        # data = data.view(-1, 768)
        # if self.transform is not None:
        #     data = self.transform(data)
        return data, targets



class ClassRoom_ZLJ(Dataset):
    """
    一个用于加载不同类型小样本和连续学习数据集的类。
    """
    
    def __init__(self, fscil = True, train = True, classroom_type = 'resnet_a1', session = 0, f_shot = 5):
        self.train = train
        # 根据参数选择预处理模型和增量类型
        self.pre_model, cil_type = classroom_type.strip().rsplit('_', 1)           
        self.curr_class = []
        self.kown_class = 0
        self.num_max = 50000000 #设置小样本训练阶段样本数量，和连续学习的样本数量
        self.classes = classes_ZLJ
        self.noise_level = 0.1
        import clip
        temp_model, train_transforms, val_transforms = clip.load("ViT-B/16", device='cuda', jit=False)
        del temp_model        
        if train==True:
            self.transform = transforms.Compose(train_transforms)
        else:
            self.transform= transforms.Compose(val_transforms)
        
        
        
    
        #输入检查
        if self.pre_model not in ['resnet', 'vit', 'clipvit']:
            raise TypeError("预训练模型类型选择错误，可选[resnet、vit、clipvit]")
        if cil_type not in ['a1', 'a2', 'b1', 'b2']:
            raise TypeError("增量模式选择错误，可选[a1、a2、b1、b2]")

        #根据是否小样本，设置加载数据上限
        if fscil == True:
            if session > 0:
                self.num_max = f_shot
            class_list = fscil_class
        else:
            class_list = cil_class
        
        #根据选择预处理模型类型，设置不同的样本根目录
        root_base = '/media/dataset/dataset_36453_raw/image_features'
        self.root = os.path.join(root_base, self.pre_model)
            
        if self.train:
            if f_shot == 5:
                file_path = './dataset_reqs/FSCIL/classroom/b2/classroom_train.txt'
            else:
                file_path = './dataset_reqs/FSCIL/classroom_'+str(f_shot)+'shot/b2/classroom_train.txt'
        else:
            if f_shot == 5:
                file_path = './dataset_reqs/FSCIL/classroom/b2/classroom_test.txt'
            else:   
                file_path = './dataset_reqs/FSCIL/classroom_'+str(f_shot)+'shot/b2/classroom_test.txt'
        all_datas, all_targets = [], []

        with open(file_path, "r") as f:
            for line in f:
                split_line = line.split(" ")
                path = split_line[0].strip()
                # x.append(os.path.join(self.data_path, path))
                all_datas.append(path)
                all_targets.append(int(split_line[-1].strip()))
        self.all_datas = np.array(all_datas)
        self.all_targets = np.array(all_targets)
        #根据不同的加载数据类设置，计算当前seesion的旧类数量
        #根据是否 train ，设置 txt 的文件目录
        #self.txt_path = '/amax/2020/gkl/CIL/make_data/classroom/index'   
        self.txt_path = '/amax/2020/qyl/PriViLege/data/index_list/classroom/clean/index'
        index = ['a1', 'a2', 'b1', 'b2'].index(cil_type)
        self.curr_class = class_list[index][session]
        if train:
            self.end = 'train/train_'
            if session > 0:
                if fscil == False and cil_type in ['a1', 'b1']:
                    self.kown_class = 8 + (session-1)*len(self.curr_class)
                if fscil == False and cil_type in ['a2', 'b2']:
                    self.kown_class = 12 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a1', 'b1']:
                    self.kown_class = 16 + (session-1)*len(self.curr_class)
                if fscil == True and cil_type in ['a2', 'b2']:
                    self.kown_class = 20 + (session-1)*len(self.curr_class)
        else:
            if session != 0:
                temp = []
                for i in range(0, session):
                    temp += class_list[index][i]
                self.curr_class = temp + self.curr_class   
            self.end = 'test/test_'    
            self.num_max = 100000
                  
                  
        # 当前增量阶段，单独保存每个类的数据路径和标签
        self.class_data_path = [[] for i in range(len(self.curr_class))]
        self.class_label = [[] for i in range(len(self.curr_class))]
        self.get_every_class_data()       
         
        # 一个增量阶段的总的数据路径和标签
        self.data_path = []
        self.targets = []
        self.get_all_class_data()


    def get_every_class_data(self,):
        """
        分别读取当前阶段每个类的数据和标签
        """
        for class_index in self.curr_class:
            for i in range(len(self.all_datas)):
                data_path = self.all_datas[i]
                target = self.all_targets[i]
                index = self.curr_class.index(class_index)   
                if index + self.kown_class == target and len(self.class_data_path[index]) < self.num_max:
                    self.class_data_path[index].append(data_path)
                    self.class_label[index].append(index + self.kown_class)

            # #存储每个类的 txt 文件路径
            # class_path = os.path.join(self.txt_path , (self.end + f'{class_index}.txt'))
            # with open(class_path, 'r') as file:
            #     lines = file.readlines()
            # unordered_index = list(range(len(lines)))# 将读取的数据打乱，防止顺序加载    
            # random.shuffle(unordered_index)
            
            # index = self.curr_class.index(class_index)            
            # for i in unordered_index:
            #     data_path, label = lines[i].strip().rsplit(' ', 1) # 去掉行末的换行符并分割
            #     data_path = os.path.join(self.root, data_path)
            #     label = int(label)
            #     if(self.pre_model == 'clipvit'):
            #         data_path = data_path.replace('.pkl', '_img.pkl')# 修改路径后缀
            #     if len(self.class_data_path[index]) < self.num_max:#当前数据量小于设定值, 继续向各类别列表中添加
            #         self.class_data_path[index].append(data_path)
            #         self.class_label[index].append(index + self.kown_class)
            #     else:
            #         break
            
              
    def get_all_class_data(self, ):
        """
        整合当前增量阶段所有类的数据和标签
        """
        for class_index in range(len(self.curr_class)):
            for data in zip( self.class_data_path[class_index], self.class_label[class_index] ):
                data_path, label = data
                self.data_path.append(data_path)  
                self.targets.append(label)
                
    def add_noise(self,features, noise_std=0.1):
        noise = torch.randn_like(features) * noise_std
        return features + noise
                
    
    
    
        
    def random_crop_and_resize(self,embedding_map, crop_size=[12,12], target_size=[14,14]):
        """
        对二维嵌入网格随机裁剪并调整回原始大小。
        :param embedding_map: 嵌入网格 (H', W', D)，例如 (14, 14, 768)
        :param crop_size: 裁剪大小 (crop_h, crop_w)
        :param target_size: 调整后的目标大小 (target_h, target_w)
        :return: 调整尺寸后的嵌入网格
        """
        H, W, D = embedding_map.shape
        crop_h, crop_w = crop_size
        target_h, target_w = target_size

        if crop_h > H or crop_w > W:
            raise ValueError("裁剪尺寸不能超过嵌入网格的大小")

        # 随机选择裁剪的起始点
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # 裁剪操作
        cropped_map = embedding_map[top:top + crop_h, left:left + crop_w, :]

        # 调整回原始大小
        # torch.nn.functional.interpolate 要求输入为 4D 张量，因此需要添加批量维度
        cropped_map = cropped_map.permute(2, 0, 1).unsqueeze(0)  # (D, crop_h, crop_w) -> (1, D, crop_h, crop_w)
        resized_map = F.interpolate(cropped_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
        resized_map = resized_map.squeeze(0).permute(1, 2, 0)  # 恢复到 (target_h, target_w, D)

        return resized_map

    def random_flip(self,embedding_map, horizontal=True, vertical=False):
        """
        对二维嵌入网格随机翻转。
        :param embedding_map: 嵌入网格 (H', W', D)，例如 (14, 14, 768)
        :param horizontal: 是否随机水平翻转
        :param vertical: 是否随机垂直翻转
        :return: 翻转后的嵌入网格
        """
        if horizontal and random.random() > 0.5:
            embedding_map = torch.flip(embedding_map, dims=[1])  # 水平翻转
        if vertical and random.random() > 0.5:
            embedding_map = torch.flip(embedding_map, dims=[0])  # 垂直翻转
        return embedding_map

                
    def __len__(self):
        return len(self.data_path)


    def __getitem__(self, i):
        path, targets = self.data_path[i], self.targets[i]
        data = self.transform.transforms(Image.open(path).convert("RGB"))
        
        # with open(path, 'rb') as file:
        #     data = torch.squeeze(torch.load(file)).detach().cpu()
            
        # data = data.view(14, 14, 768)
        # data = self.random_crop_and_resize(data )
        # data = self.random_flip(data )
        # data = data.view(-1, 768)
        # if self.transform is not None:
        #     data = self.transform(data)
        return data, targets



# 使用样例
if __name__ == '__main__':
      
    # 测量训练集加载时间
    train_start_time = time.time()  
      
    #train
    # for i in range(5):
    fs_trainset = ClassRoom(fscil=True, train=True, classroom_type = 'vit_a2', session = 1, f_shot= 5)
    #     print(len(fs_trainset))
    fs_testset = ClassRoom(fscil=True, train=False, classroom_type = 'vit_a2', session = 1, f_shot= 5)
    #     print(len(fs_testset))
    trainloader = DataLoader(dataset=fs_trainset, batch_size=64, 
                                    shuffle=False, num_workers=0, pin_memory=True)
    # for sample,label in fs_trainset :
    #     train_data, train_label = sample ,label
        
    for batch_idx, (data, labels) in enumerate(trainloader):
        train_data, train_label = data, labels
            
    # 测量测试集加载时间
    # train_end_time = time.time()
    # train_total_time = train_end_time - train_start_time 
    
    # # 测量测试集加载时间
    # test_start_time = time.time()
    # # test
    # for i in range(5):
    #     fs_testset = ClassRoom(fscil=True, train=False, classroom_type = 'resnet_a1', session = i, f_shot= 5)
    #     testloader = DataLoader(dataset=fs_testset, batch_size=64, 
    #                             shuffle=False, num_workers=8)
    #     for sample in testloader:
    #         test_data, test_label = sample
    
    # test_end_time = time.time()
    # test_total_time = test_end_time - test_start_time

    # print(f"Training data loading time: {train_total_time} seconds")
    # print(f"Test data loading time: {test_total_time} seconds")
    # print(f"Total time: {train_total_time + test_total_time} seconds")
        
        
# # 分别读取当前阶段每个类的数据和标签
# def get_every_class_data(self,):
    
#     with open(self.root, 'r') as file: # 打开a.txt文件
#         lines = file.readlines()
        
#     unordered_index = list(range(len(lines)))# 将读取的数据打乱，防止顺序加载    
#     random.shuffle(unordered_index)
#     for index in unordered_index:
#         data_path, label = lines[index].strip().rsplit(' ', 1) # 去掉行末的换行符并分割
#         data_path = data_path.replace("resnet", "/media/amax/82C272A7C2729EDB/dataset_36453/image_features/resnet", 1)
#         label = int(label)
        
#         if label in  self.curr_class:#如果数据类别在当前索引类中
#             index = self.curr_class.index(label)
#             if len(self.class_data_path[index]) < self.num_max:#当前数据量小于设定值, 继续向各类别列表中添加
                
#                 # if(self.pre_model == 'clipvit'):
#                 #     data_path = data_path.replace('.pkl', '_img.pkl')# 修改路径后缀
#                 #     data_path = data_path.replace('resnet', 'clipvit')# 修改路径后缀
#                 # if(self.pre_model == 'vit'):
#                 #     data_path = data_path.replace('resnet', 'vit')# 修改路径后缀   
#                 self.class_data_path[index].append(data_path)
#                 self.class_label[index].append(index + self.kown_class)
    