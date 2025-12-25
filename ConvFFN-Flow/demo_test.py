import sys
sys.path.append('core')  # 将'core'目录添加到系统路径，以便导入模块
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse  # 用于命令行参数解析
import os  # 用于操作系统相关功能
import numpy as np  # NumPy库，用于数组操作
import torch  # PyTorch库，用于深度学习
from PIL import Image  # 用于加载和处理图像
#from core.raft import RAFT  # 导入RAFT模型
from core.raft_convext import RAFT  # 导入RAFT模型

from core.utils.utils import InputPadder  # 导入用于输入数据填充的工具类
import struct
import glob
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
'''

******************************************************实测实验推理






'''

def is_valid_folder(folder_name):
    """判断是否是符合 *-* 格式的文件夹名"""
    return '-' in folder_name and all(part.isdigit() for part in folder_name.split('-'))
def load_image(imfile):
    # 加载图像并转换为NumPy数组，像素值类型为uint8
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # 检查图像维度：如果是灰度图（H, W），扩展为伪彩色（H, W, 3）
    if img.ndim == 2:  # 灰度图
        img = np.stack([img] * 3, axis=-1)  # 转换为伪彩色
    # 转换为PyTorch张量，并调整维度顺序为(C, H, W)，同时将像素值转换为浮点型
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # 将图像添加batch维度并移动到指定设备上

    return img[None].to(DEVICE)
def save_flow(filename, flow):
    """
    保存光流数据为.flo格式文件
    参数:
    filename: 输出文件路径
    flow: 光流数据，形状为 (1, 2, H, W)，存储了光流的 (u, v) 分量
    """
    # 调整维度顺序为 (H, W, 2)
    flow = flow.squeeze(0).transpose((1, 2, 0))
    h, w = flow.shape[:2]

    with open(filename, 'wb') as f:
        # 写入魔术数字，使用系统默认字节序
        f.write(struct.pack('f', 202021.25))
        # 写入宽度，使用系统默认字节序
        f.write(struct.pack('i', w))
        # 写入高度，使用系统默认字节序
        f.write(struct.pack('i', h))
        # 确保数据是 np.float32 类型，展平并确保内存连续性
        flow = np.ascontiguousarray(flow.astype(np.float32)).flatten()
        f.write(flow.tobytes())
def viz(img, flo,output_path,step=15):
    # 调整图像的维度顺序，从(C, H, W)变为(H, W, C)，并转换为NumPy数组
    img = img[0].permute(1, 2, 0).cpu().numpy()
    # 同样处理光流数据
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # 计算光流的幅值（magnitude）
    u = flo[:, :, 0]
    v = flo[:, :, 1]
    magnitude = np.sqrt(u**2 + v**2)
    h, w = magnitude.shape
    x, y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))

    u_sampled = u[::step, ::step]
    v_sampled = v[::step, ::step]
    # 自动计算颜色条的最大最小值
    # vmin = np.min(velocity)
    # vmax = np.max(velocity)
    vmin_pix = np.min(magnitude)
    vmax_pix = np.max(magnitude)
    plt.figure()
    # im = plt.imshow(magnitude, cmap='jet',vmin=vmin, vmax=5)
    im = plt.imshow(magnitude, cmap='jet', vmin=2, vmax=vmax_pix)
    plt.quiver(x, y, u_sampled, v_sampled, angles='xy', scale_units='xy', scale=0.5, color='white')
    cbar = plt.colorbar(im, shrink=0.6)
    # cbar.set_label('Water Flow Velocity (m/s)')
    cbar.set_label('Optical Flow Magnitude(pix/frame)')
    # 隐藏坐标轴的刻度
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    plt.title("RAFT ")
    plt.savefig(output_path, bbox_inches='tight', dpi=600)  # 这个是保存图片
    #plt.show()
    plt.close(1)
from tqdm import tqdm
def demo(args):
    # 初始化并加载RAFT模型
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, weights_only=True), strict=False)
    model = model.module.to(DEVICE).eval()

    # 创建输出路径
    os.makedirs(args.output_path, exist_ok=True)

    # 获取所有符合 *-* 的子目录
    all_folders = [f for f in os.listdir(args.path)
                   if os.path.isdir(os.path.join(args.path, f)) and is_valid_folder(f)]
    all_folders = sorted(all_folders)

    for folder in tqdm(all_folders, desc="文件夹处理进度", ncols=80):
        folder_path = os.path.join(args.path, folder)
        output_folder = os.path.join(args.output_path, folder)
        os.makedirs(output_folder, exist_ok=True)

        # 获取当前子目录下所有图像
        images = sorted(glob.glob(os.path.join(folder_path, '*.png')) +
                        glob.glob(os.path.join(folder_path, '*.jpg')) +
                        glob.glob(os.path.join(folder_path, '*.jpeg')) +
                        glob.glob(os.path.join(folder_path, '*.bmp')) +
                        glob.glob(os.path.join(folder_path, '*.tiff')) +
                        glob.glob(os.path.join(folder_path, '*.gif')))

        if len(images) < 2:
            print(f"[警告] 跳过文件夹 {folder}，因为图像数量不足。")
            continue

        for i in tqdm(range(len(images) - 1), desc=f"[{folder}] 图像处理", leave=False, ncols=70):
            imfile1 = images[i]
            imfile2 = images[i + 1]

            # 加载图像并进行填充
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, mode='sintel')
            image1, image2 = padder.pad(image1, image2)

            # 计算光流
            with torch.no_grad():
                _, flow_up = model(image1, image2, iters=12, test_mode=True)
                flow_up = padder.unpad(flow_up)

            # 输出文件名（基于第一帧命名）
            basename = os.path.splitext(os.path.basename(imfile1))[0]
            flo_filename = os.path.join(output_folder, f"{basename}.flo")
            image_output = os.path.join(output_folder, f"{basename}.png")

            # 保存结果
            save_flow(flo_filename, flow_up.cpu().numpy())
            viz(image1, flow_up, image_output)

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='check/250000_raft-convext_JIARUO.pth', help="restore checkpoint")  # 模型权重路径

    #E:\test_data\test_river\UAV\Area2\20_river_cai;checkpoints/250000_raft-river.pth;check/250000_raft-convext_new.pth
    parser.add_argument('--path',default='E:/CRG/',help="dataset for evaluation")  # 数据集路径
    parser.add_argument('--small', action='store_true', help='use small model')  # 是否使用小模型
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')  # 是否使用混合精度
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficient correlation implementation')  # 是否使用高效相关性实现
    parser.add_argument('--output_path',default='E:/CRG_demo/convfft_nopos', help="output path to save optical flow images")  # 输出路径
    parser.add_argument('--use_basic_layer', action='store_true', default=False, help='Whether to use the basic layer')
    args = parser.parse_args()  # 解析命令行参数

    demo(args)  # 调用主函数
