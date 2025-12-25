import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import core.datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.raft_convext import RAFT
from core.utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = core.datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results
def validate_river_data(model, iters=12):
    """
    使用 river_data数据集的 `test` 文件夹对模型进行评估。
    Args:
        model: 要评估的光流模型
        test_path: `test` 文件夹的路径
        iters: 迭代次数
    Returns:
        包含平均 EPE 和像素准确率的结果字典
    """
    model.eval()
    results = {}  # 存储评估结果

    val_dataset = core.datasets.RiverDataset(split='Validation', )#Validation;ceshi
    # 加载 Sintel 训练集对应数据类型的数据集，dstype 代表数据类型（'clean' 或 'final'）。
    for val_id in tqdm(range(len(val_dataset)), desc="Validation"):
        epe_list = []
        # 遍历验证集中的每个样本（图像对）。
        image1, image2, flow_gt, _ = val_dataset[val_id]
        # ===================== 后续处理保持不变 =====================
        image1 = image1[None].cuda()  # 添加batch维度 [1, C, 393, 392]
        image2 = image2[None].cuda()
        # 从数据集中加载图像对（image1 和 image2）及其对应的真实光流（flow_gt）。
        # 为图像添加一个 batch 维度，并将图像移至 GPU。
        padder = InputPadder(image1.shape)
        # 创建一个 InputPadder 对象，用于处理图像的填充（以适应网络输入的大小要求）。
        image1, image2 = padder.pad(image1, image2)
        # 对图像进行填充，以确保它们的尺寸可以被网络处理。
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        # 使用模型进行前向传播，计算预测的光流（flow_pr）。`iters` 表示迭代次数，`test_mode=True` 表示模型在推理模式下运行。
        flow = padder.unpad(flow_pr[0]).cpu()
        # 移除填充并将预测的光流转换为 CPU 张量。
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        # 计算 End Point Error（EPE），即预测光流和真实光流之间的欧几里得距离。
        epe_list.append(epe.view(-1).detach().numpy())
        # 将 EPE 展平成一维数组并转换为 NumPy 数组，添加到 epe_list 中。
    epe_all = np.concatenate(epe_list)
    # 将所有图像对的 EPE 合并为一个长数组。
    epe = np.mean(epe_all)
   # print('epe_all', epe)
    # 计算所有图像对的平均 EPE。
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)
    # 打印评估结果
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, px1, px3, px5))
    # 将结果整合成字典返回
    results = {'epe': epe, 'px1': px1, 'px3': px3, 'px5': px5}
    # 将当前数据类型的平均 EPE 存入结果字典。
    return results
def validate_river_data1(model, iters=12):
    """
    使用 river_data 数据集的 Validation 文件夹对模型进行评估。
    按照类别（Laminar、Structured、Tracer、Turbulent）分别统计 EPE 和像素精度。

    Args:
        model: 要评估的光流模型
        iters: 推理迭代次数

    Returns:
        dict: 每个类别及整体的评估结果，包括 EPE、1px、3px、5px 的统计
    """
    model.eval()
    results = {}  # 存储评估结果

    val_dataset = core.datasets.RiverDataset(split='Validation')
    dataset_size = len(val_dataset)
    print(f"验证集大小: {dataset_size} 样本")

    category_epe_list = {}

    # 使用 tqdm 显示进度条
    for val_id in tqdm(range(dataset_size), desc="Validation"):
        image1, image2, flow_gt, extra_info = val_dataset[val_id]

        # 获取路径并提取类别名
        path = val_dataset.flow_list[val_id]
        category_name = path.split(os.sep)[-3]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # 推理模型
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        # 计算 EPE
        epe = torch.norm(flow - flow_gt, dim=0)
        epe_flat = epe.view(-1).detach().numpy()

        # 分类统计
        if category_name not in category_epe_list:
            category_epe_list[category_name] = []
        category_epe_list[category_name].append(epe_flat)

    # 汇总每类结果
    all_epe_flat = []
    for category, epe_arrays in category_epe_list.items():
        epe_all = np.concatenate(epe_arrays)
        all_epe_flat.append(epe_all)
        mean_epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(f"[{category}] EPE: {mean_epe:.4f}, 1px: {px1:.4f}, 3px: {px3:.4f}, 5px: {px5:.4f}")
        results[category] = {'epe': mean_epe, 'px1': px1, 'px3': px3, 'px5': px5}

    # 总体评估
    total_epe = np.concatenate(all_epe_flat)
    total_mean_epe = np.mean(total_epe)
    total_px1 = np.mean(total_epe < 1)
    total_px3 = np.mean(total_epe < 3)
    total_px5 = np.mean(total_epe < 5)

    print(f"[Overall] EPE: {total_mean_epe:.4f}, 1px: {total_px1:.4f}, 3px: {total_px3:.4f}, 5px: {total_px5:.4f}")
    results['Overall'] = {'epe': total_mean_epe, 'px1': total_px1, 'px3': total_px3, 'px5': total_px5}

    # 如果需要保存结果到文件，可以取消下面的注释：
    # import json
    # with open("river_val_results.json", "w") as f:
    #     json.dump(results, f, indent=2)

    return results
@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='check/250000_raft-convext_JIARUO.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='River_data', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()
    # 添加命令行参数，用于指定模型路径、评估数据集、是否使用小型模型等选项。checkpoints/sun_sunRiver_250000.pth才是我的换掉头部的特征模型

    model = torch.nn.DataParallel(RAFT(args))
    checkpoint = torch.load(args.model)  # 加载完整检查点

    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    # 加载模型权重。
    # 将模型移至 GPU 并设置为评估模式。
    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)
    # 创建 Sintel 和 KITTI 数据集的提交文件（注释掉了）。
    with torch.no_grad():
        # 禁用梯度计算，进行模型推理。
        if args.dataset == 'River_data':
            validate_river_data1(model.module)
        # 如果评估数据集是 'chairs'，则进行 Chairs 数据集验证。
        elif args.dataset == 'sintel':
            validate_sintel(model.module)
        # 如果评估数据集是 'sintel'，则进行 Sintel 数据集验证。
        elif args.dataset == 'kitti':
            validate_kitti(model.module)
        # 如果评估数据集是 'kitti'，则进行 KITTI 数据集验证。


