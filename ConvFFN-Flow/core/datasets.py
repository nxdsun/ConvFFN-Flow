# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from tqdm import tqdm

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        flow = np.nan_to_num(flow, nan=0.0)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='F:/datasets/FlyingChair/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='F:/datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='F:/datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
#########################这个是读取未分类的结果
class RiverDataset(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='F:/sun/data/deepflow_small', sparse=False):
        """
        deepflow_small

数据集结构：
    deepflow_small/
    ├── training/
    │   ├── 5cai/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   ├── flow.flo
    │   │   └── ...
    │   ├── 6cai/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   ├── flow.flo
    │   │   └── ...
    │   ├── 7cai/
    │   └── 8cai/
    ├── validation/
      │   ├── 5cai/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   ├── flow.flo
    │   │   └── ...
    │   ├── 6cai/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   ├── flow.flo
    │   │   └── ...
    │   ├── 7cai/
    │   └── 8cai/
        这个是加载原来一级文件的版本
        继承自 FlowDataset，只需改动数据加载逻辑部分
        """
        super(RiverDataset, self).__init__(aug_params, sparse)  # 调用父类 FlowDataset 的构造函数

        self.root = root  # 数据集根目录
        self.split = split  # 数据集划分方式

        kind_root = osp.join(root, split)  # 获取数据集的子文件夹路径
        self.kinds = os.listdir(kind_root)  # 获取所有类别文件夹（例如 1, 2）

        for kind in self.kinds:
            kind_path = osp.join(kind_root, kind)
            images = sorted(glob(osp.join(kind_path, '*.jpg')))  # 获取所有图像路径
            flows = sorted(glob(osp.join(kind_path, '*.flo')))  # 获取所有光流文件路径

            # 确保 flows 的数量与 images 数量一致（每两个图像对应一个光流文件）
            for i in range(0, len(images), 2):  # 每次取两张图像
                if i + 1 < len(images):  # 确保有两个图像
                    image_pair = [images[i], images[i + 1]]  # 将两个图像文件作为一对存储
                    self.image_list.append(image_pair)

                    flow_index = i // 2  # 对应的光流文件的索引
                    if flow_index < len(flows):
                        self.flow_list.append(flows[flow_index])  # 添加对应的光流文件
                    else:
                        print(f"Warning: Missing flow file for image pair: {images[i]}, {images[i + 1]}")
class RiverDataset(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='F:/sun/data/deepflow_small', sparse=False):
        """deepflow_small;deepflow_mini、deepflowyuan
        farnback；TVL1；widim
        数据集结构：

        deepflow_small/
    ├── training/
    │   ├── Laminar/
    │   │   ├── 5cai/
    │   │   │   ├── img1.jpg
    │   │   │   ├── img2.jpg
    │   │   │   ├── flow.flo
    │   │   │   └── ...
    │   ├── Structured/
    │   ├── Tracer/
    │   └── Turbulent/
    ├── validation/
    │   ├── Laminar/
    │   │   ├── 5cai/
    │   │   │   ├── img1.jpg
    │   │   │   ├── img2.jpg
    │   │   │   ├── flow.flo
    │   │   │   └── ...
    │   ├── Structured/
    │   ├── Tracer/
    │   └── Turbulent/
        继承自 FlowDataset，只需改动数据加载逻辑部分
        """
        super(RiverDataset, self).__init__(aug_params, sparse)  # 调用父类 FlowDataset 的构造函数

        self.root = root  # 数据集根目录
        self.split = split  # 数据集划分方式

        kind_root = osp.join(root, split)  # 获取数据集的子文件夹路径
        self.kinds = os.listdir(kind_root)  # 获取所有类别文件夹（例如 1, 2）

        # 修改点1：遍历两层目录结构
        self.kinds = []
        for top_folder in os.listdir(kind_root):  # 第一层目录（如Laminar）
            top_path = osp.join(kind_root, top_folder)
            if osp.isdir(top_path):
                # 获取第二层子目录（如5cai）
                for sub_folder in os.listdir(top_path):
                    sub_path = osp.join(top_path, sub_folder)
                    if osp.isdir(sub_path):
                        self.kinds.append(sub_path)  # 存储完整子目录路径

        # 修改点2：直接遍历最终的数据目录
        for data_dir in self.kinds:  # 每个data_dir已经是类似 .../Laminar/5cai 的路径
            images = sorted(glob(osp.join(data_dir, '*.jpg')))
            flows = sorted(glob(osp.join(data_dir, '*.flo')))
            # 确保 flows 的数量与 images 数量一致（每两个图像对应一个光流文件）
            for i in range(0, len(images), 2):  # 每次取两张图像
                if i + 1 < len(images):  # 确保有两个图像
                    image_pair = [images[i], images[i + 1]]  # 将两个图像文件作为一对存储
                    self.image_list.append(image_pair)

                    flow_index = i // 2  # 对应的光流文件的索引
                    if flow_index < len(flows):
                        self.flow_list.append(flows[flow_index])  # 添加对应的光流文件
                        # # 🔥 这里加上打印，检查每一对匹配情况
                        # print(
                        #     f"Image pair: {osp.basename(images[i])}, {osp.basename(images[i + 1])} --> Flow: {osp.basename(flows[flow_index])}")

                    else:
                        print(f"Warning: Missing flow file for image pair: {images[i]}, {images[i + 1]}")
            # 打印当前的图像文件名和光流文件名
               # print(f"Image pair: {image_pair}, Flow file: {flows[flow_index]}")



def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})

            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things
    elif args.stage == 'River_data':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}  # 49*63
        train_dataset = RiverDataset(aug_params, split='training')
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

