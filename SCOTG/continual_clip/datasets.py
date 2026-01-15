import os
import torch.nn as nn
import torch
from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50,_ContinuumDataset
)
from .utils import get_dataset_class_names
from continuum.tasks import TaskSet, TaskType
from continuum.tasks.image_path_task_set import PathTaskSet
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from torchvision import datasets as torchdata
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import json


class FSCILPathTaskSet(PathTaskSet):
    """A task dataset returned by the CLLoader specialized into array of image's path to images.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param target_trsf: The transformations to apply on the labels.
    :param bounding_boxes: The bounding boxes annotations to crop images
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
            bounding_boxes: Optional[np.ndarray] = None
    ):
        super().__init__(x, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]
        x = Image.open(x).convert("RGB")
        return x
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.bounding_boxes is not None:
            bbox = self.bounding_boxes[index]
            x = x.crop((
                max(bbox[0], 0),  # x1
                max(bbox[1], 0),  # y1
                min(bbox[2], x.size[0]),  # x2
                min(bbox[3], x.size[1]),  # y2
            ))

        x, y, t = self._prepare_data(x, y, t)

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        path = os.path.basename(self._x[index])
        return x, y, t, path

class MMEADataset(ImageNet100):
    """Continuum dataset for datasets with tree-like structure.

    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self, *args, data_subset: Union[Tuple[np.array, np.array], str, None] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_subset = data_subset


    def _parse_subset(
            self,
            subset: Union[Tuple[np.array, np.array], str, None],
            train: bool = True
    ) -> Tuple[np.array, np.array]:
        if isinstance(subset, str):
            x, y = [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ")
                    path = split_line[0].strip()
                    # x.append(os.path.join(self.data_path, path))
                    x.append(path)
                    y.append(int(split_line[-1].strip()))
            x = np.array(x)
            y = np.array(y)
            return x, y
        return subset  # type: ignore

    def _download(self):
        pass

class ClassIncremental_MMEA(ClassIncremental):
    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        class_order: Union[List[int], None]=None
    ) -> None:

        self.cl_dataset = cl_dataset
        self.increment = increment
        self.initial_increment = initial_increment
        self.class_order = class_order

        self._nb_tasks = self._setup(nb_tasks)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=self._nb_tasks, transformations=transformations)

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return MMEATaskset(
            x=x, y=y, t=t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf = None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )

class MMEATaskset(PathTaskSet):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
            bounding_boxes: Optional[np.ndarray] = None
    ):
        super().__init__(x, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x=[]
        dir_name = self._x[index]
        idx = 0
        for file_name in os.listdir(str(dir_name)):
            if file_name.split('_')[0]=='img':
                if idx%15==0:
                    x.append(Image.open(os.path.join(dir_name,file_name)).convert("RGB"))
                idx=idx+1
        return x  

    def _prepare_data(self, x, y, t):
        if self.trsf is not None:
            out = []
            for frame in x:
                out.append(self.get_task_trsf(t)(frame).unsqueeze(0))
            x = torch.cat(out,dim=0)
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, t
    
class ClassIncremental_classroom(ClassIncremental):
    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        class_order: Union[List[int], None]=None
    ) -> None:

        # self.cl_dataset = cl_dataset
        # self.increment = increment
        # self.initial_increment = initial_increment
        # self.class_order = class_order

        # self._nb_tasks = self._setup(nb_tasks)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_tasks, increment = increment, initial_increment=initial_increment, transformations=transformations,class_order=class_order)

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return PathTaskSet(
            x=x, y=y, t=t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf = None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )



 
class FSCIL_classroom(ClassIncremental):
    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        class_order: Union[List[int], None]=None
    ) -> None:

        # self.cl_dataset = cl_dataset
        # self.increment = increment
        # self.initial_increment = initial_increment
        # self.class_order = class_order

        # self._nb_tasks = self._setup(nb_tasks)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_tasks, increment = increment, initial_increment=initial_increment, transformations=transformations,class_order=class_order)

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        # return PathTaskSet(
        #     x=x, y=y, t=t,
        #     trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
        #     target_trsf = None,
        #     bounding_boxes=self.cl_dataset.bounding_boxes,
        # )
        return FSCILPathTaskSet(
            x=x, y=y, t=t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf = None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )

 
class FSCIL_classroom_llava(ClassIncremental):
    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        class_order: Union[List[int], None]=None
    ) -> None:

        # self.cl_dataset = cl_dataset
        # self.increment = increment
        # self.initial_increment = initial_increment
        # self.class_order = class_order

        # self._nb_tasks = self._setup(nb_tasks)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_tasks, increment = increment, initial_increment=initial_increment, transformations=transformations,class_order=class_order)

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return PathTaskSet_llava(
            x=x, y=y, t=t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf = None,
            bounding_boxes=self.cl_dataset.bounding_boxes,
        )


class PathTaskSet_llava(PathTaskSet):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
            bounding_boxes: Optional[np.ndarray] = None
    ):
        llava_root = '/data_25T/zlj/FSCIL/data/ARIC_visual_filter/description/description_llava.json'
        with open(llava_root,'r') as f:
            self.llava=json.load(f)
        super().__init__(x, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        img_path = self._x[index]
        text = self.llava[img_path]
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.bounding_boxes is not None:
            bbox = self.bounding_boxes[index]
            x = x.crop((
                max(bbox[0], 0),  # x1
                max(bbox[1], 0),  # y1
                min(bbox[2], x.size[0]),  # x2
                min(bbox[3], x.size[1]),  # y2
            ))

        x, y, t = self._prepare_data(x, y, t)

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        return x, y, t,text



def _handle_negative_indexes(index: int, total_len: int) -> int:
    if index < 0:
        index = index % total_len
    return index


def sample_images(x, y, t, num_samples=5):
    # 字典存储每个标签对应的索引
    label_to_indices = defaultdict(list)
    
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)
    
    # 从每个标签的索引中随机抽取 num_samples 个
    sampled_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) > num_samples:
            sampled_indices.extend(np.random.choice(indices, num_samples, replace=False))
        else:
            sampled_indices.extend(indices)
    
    # 打乱抽样结果
    np.random.shuffle(sampled_indices)
    return sampled_indices


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


def get_dataset(cfg, is_train,mode, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes

    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        data_path = os.path.join(cfg.dataset_root, "ImageNet")
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join('/data_25T/zlj/FSCIL/ZSCL-main/cil/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
            # data_subset=os.path.join('./Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]

    elif cfg.scenario == "class":
        if cfg.dataset == "mmea":
            data_path = None
            if is_train:
                data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea/mydataset_train.txt'
                # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_train.txt'
            else:
                if mode=='val':
                    data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea/mydataset_val.txt'
                    # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_val.txt'
                elif mode=='test':
                    data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea/mydataset_test.txt'
                    # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_test.txt'

            dataset = MMEADataset(
                data_path, 
                train=is_train,
                data_subset=data_subset
                # data_subset=os.path.join('./Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
            )
            classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

        elif cfg.dataset == "classroom":
            data_path = None
            if is_train:
                data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/classroom/classroom_train.txt'
                # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_train.txt'
            else:
                data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/classroom/classroom_test.txt'

            dataset = MMEADataset(
                data_path, 
                train=is_train,
                data_subset=data_subset
                # data_subset=os.path.join('./Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
            )
            classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.scenario == "fscil":
        if cfg.dataset == "mmea":
            data_path = None
            if is_train:
                data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/FSCIL/mmea/mydataset_train.txt'
                # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_train.txt'
            else:
                if mode=='val':
                    data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/FSCIL/mmea//mydataset_val.txt'
                    # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_val.txt'
                elif mode=='test':
                    data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/FSCIL/mmea//mydataset_test.txt'
                    # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_test.txt'

            dataset = MMEADataset(
                data_path, 
                train=is_train,
                data_subset=data_subset
                # data_subset=os.path.join('./Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
            )
            classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

        elif cfg.dataset == "classroom":
            data_path = None
            if is_train:
                data_subset=os.path.join(cfg.train_file_path,'classroom_train.txt')
                # data_subset='/data_25T/zlj/FSCIL/ZSCL-main/cil_me/dataset_reqs/mmea_s/mydataset_train.txt'
            else:
                data_subset=os.path.join(cfg.train_file_path,'classroom_test.txt')

            dataset = MMEADataset(
                data_path, 
                train=is_train,
                data_subset=data_subset
                # data_subset=os.path.join('./Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
            )
            classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, mode,transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train,mode)

    if cfg.scenario == "class":
        if cfg.dataset == 'mmea':
            scenario = ClassIncremental_MMEA(
                dataset,
                initial_increment=cfg.initial_increment,
                increment=cfg.increment,
                transformations=transforms.transforms, # Convert Compose into list
                class_order=cfg.class_order,
            )
        elif cfg.dataset == 'classroom':
            scenario = ClassIncremental_classroom(
                dataset,
                initial_increment=cfg.initial_increment,
                increment=cfg.increment,
                transformations=transforms.transforms, # Convert Compose into list
                class_order=cfg.class_order,
            )
    elif cfg.scenario == "fscil":
        if cfg.method =='cpe_llava':
            if cfg.dataset == 'mmea':
                scenario = ClassIncremental_MMEA(
                    dataset,
                    initial_increment=cfg.initial_increment,
                    increment=cfg.increment,
                    transformations=transforms.transforms, # Convert Compose into list
                    class_order=cfg.class_order,
                )
            elif cfg.dataset == 'classroom':
                scenario = FSCIL_classroom_llava(
                    dataset,
                    initial_increment=cfg.initial_increment,
                    increment=cfg.increment,
                    transformations=transforms.transforms, # Convert Compose into list
                    class_order=cfg.class_order,
                )
        else:
            if cfg.dataset == 'mmea':
                scenario = ClassIncremental_MMEA(
                    dataset,
                    initial_increment=cfg.initial_increment,
                    increment=cfg.increment,
                    transformations=transforms.transforms, # Convert Compose into list
                    class_order=cfg.class_order,
                )
            elif cfg.dataset == 'classroom':
                scenario = FSCIL_classroom(
                    dataset,
                    initial_increment=cfg.initial_increment,
                    increment=cfg.increment,
                    transformations=transforms.transforms, # Convert Compose into list
                    class_order=cfg.class_order,
                )
    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names