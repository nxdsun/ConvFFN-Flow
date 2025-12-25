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
#from core.raft_convext import RAFT  # 导入RAFT模型
from core.raft import RAFT
from core.utils.utils import InputPadder  # 导入用于输入数据填充的工具类
import struct
import glob
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def read_flow(filename):
    """读取.flo格式光流文件"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        if magic != 202021.25:
            raise Exception('Invalid .flo file')
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(w * h * 8), dtype=np.float32)
        flow = np.resize(data, (h, w, 2))
    return flow
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
    plt.title("COVNEXT")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 这个是保存图片
    #plt.show()
    plt.close(1)

from mpl_toolkits.axes_grid1 import make_axes_locatable
def save_error_heatmap(flow_pred, flow_gt, save_path, max_val=2.0):
    # 计算误差图
    error = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))

    # 创建图像
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error, cmap='copper', vmin=0)
    ax.axis('off')
    ax.set_title('Optical Flow Error Heatmap')

    # 创建等高的颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f'Flow Error (clipped at {max_val} pixels)')

    # 保存热力图
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 保存误差矩阵为 .npy 文件（与热力图名称一致）
    npy_path = os.path.splitext(save_path)[0] + '.npy'
    np.save(npy_path, error)


def get_image_pairs(path):
    # 找到所有以 _a. 结尾的文件
    images_a = sorted(glob.glob(os.path.join(path, '*_a.jpg')))
    pairs = []
    for im_a in images_a:
        im_b = im_a.replace('_a.', '_b.')
        if os.path.exists(im_b):
            pairs.append((im_a, im_b))
        else:
            print(f"Warning: 找不到对应的后帧图像 {im_b}，跳过该对。")
    return pairs
def demo(args):
    import time
    # 加载RAFT模型并启用DataParallel（支持多GPU）
    model = torch.nn.DataParallel(RAFT(args))
    # 加载模型权重，只加载权重部分
    model.load_state_dict(torch.load(args.model, weights_only=True), strict=False)
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image_pairs = get_image_pairs(args.path)

        print(f"共找到 {len(image_pairs)} 对图像进行处理。")
        # 自动创建输出文件夹（如果不存在）
        os.makedirs(args.output_path, exist_ok=True)

        total_infer_time = 0.0  # 累计推理时间
        count = 0

        for imfile1, imfile2 in image_pairs:
            print(f"处理图像对: {os.path.basename(imfile1)} 和 {os.path.basename(imfile2)}")
            image1 = load_image(imfile1).to(DEVICE)
            image2 = load_image(imfile2).to(DEVICE)

            # 填充图像到合适尺寸
            padder = InputPadder(image1.shape, mode='sintel')
            image1, image2 = padder.pad(image1, image2)

            # 只统计模型推理时间
            start_infer = time.time()
            flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 保证GPU推理完成，计时准确
            end_infer = time.time()

            infer_time = (end_infer - start_infer) * 1000  # 转换成毫秒
            total_infer_time += infer_time
            count += 1
            print(f"模型推理耗时: {infer_time:.3f} 毫秒")
            # 裁剪回原始尺寸
            flow_up = padder.unpad(flow_up)

            # 定义输出路径
            output_name = os.path.splitext(os.path.basename(imfile1))[0]
            prefix = output_name[:-2] if output_name.endswith('_a') else output_name

            flo_filename = os.path.join(args.output_path, f"flow_{prefix}.flo")
            output_path = os.path.join(args.output_path, f"flow_{prefix}.jpg")

            # 保存光流文件
            save_flow(flo_filename, flow_up.cpu().numpy())

            # 可视化光流
            viz(image1, flow_up, output_path)

            output_name = os.path.splitext(os.path.basename(imfile1))[0]  # 例如 '0001_a'
            prefix = output_name[:-2] if output_name.endswith('_a') else output_name  # 变成 '0001'

            flo_gt_path = os.path.join(args.flo_path, prefix + '.flo') if hasattr(args, 'flo_path') else os.path.join(
                args.path, prefix + '.flo')

            # 判断真实光流是否存在
            if os.path.exists(flo_gt_path):
                flow_pred = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flow_gt = read_flow(flo_gt_path)

                if flow_pred.shape == flow_gt.shape:
                    error_heatmap_path = os.path.join(args.output_path, prefix + '_heatmap.png')
                    save_error_heatmap(flow_pred, flow_gt, error_heatmap_path)
                    print(f"已生成误差热力图及误差矩阵: {error_heatmap_path} / .npy")
                else:
                    print(f"预测光流与真实光流尺寸不匹配 ({flow_pred.shape} vs {flow_gt.shape})，跳过误差热力图。")
            else:
                print(f"未找到真实光流文件 {flo_gt_path}，跳过误差热力图。")

        if count > 0:
            print(f"平均模型推理耗时: {total_infer_time / count:.3f} 毫秒")
        else:
            print("没有处理任何图像对。")

# def demo(args):
#     # 加载RAFT模型并启用DataParallel（支持多GPU）
#     model = torch.nn.DataParallel(RAFT(args))
#     # 加载模型权重，只加载权重部分
#     model.load_state_dict(torch.load(args.model, weights_only=True), strict=False)
#     model = model.module
#     model.to(DEVICE)
#     model.eval()
#
#     with torch.no_grad():
#         image_pairs = get_image_pairs(args.path)
#
#         print(f"共找到 {len(image_pairs)} 对图像进行处理。")
#         # 自动创建输出文件夹（如果不存在）
#         os.makedirs(args.output_path, exist_ok=True)
#         total_time = 0.0  # 累计时间
#         count = 0
#         for imfile1, imfile2 in image_pairs:
#             import time
#             start_time = time.time()  # 开始计时
#             print(f"处理图像对: {os.path.basename(imfile1)} 和 {os.path.basename(imfile2)}")
#             image1 = load_image(imfile1).to(DEVICE)
#             image2 = load_image(imfile2).to(DEVICE)
#
#             # 填充图像到合适尺寸
#             padder = InputPadder(image1.shape, mode='sintel')
#             image1, image2 = padder.pad(image1, image2)
#
#             # 使用RAFT模型计算光流
#             flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
#
#             # 裁剪回原始尺寸
#             flow_up = padder.unpad(flow_up)
#
#             # 定义输出路径
#             output_path = os.path.join(args.output_path, f"flow_{os.path.splitext(os.path.basename(imfile1))[0]}.png")
#             flo_filename = os.path.splitext(output_path)[0] + '.flo'
#
#             # 保存光流文件
#             save_flow(flo_filename, flow_up.cpu().numpy())
#
#             # 可视化光流
#             viz(image1, flow_up, output_path)
#             output_name = os.path.splitext(os.path.basename(imfile1))[0]  # 例如 '0001_a'
#             prefix = output_name[:-2] if output_name.endswith('_a') else output_name  # 变成 '0001'
#
#             flo_gt_path = os.path.join(args.flo_path, prefix + '.flo') if hasattr(args, 'flo_path') else os.path.join(
#                 args.path, prefix + '.flo')
#
#             # 判断真实光流是否存在
#             if os.path.exists(flo_gt_path):
#                 # 转换预测光流为 numpy 格式
#                 flow_pred = flow_up[0].permute(1, 2, 0).cpu().numpy()
#                 flow_gt = read_flow(flo_gt_path)
#
#                 if flow_pred.shape == flow_gt.shape:
#                     # 构造误差热力图保存路径
#                     error_heatmap_path = os.path.join(args.output_path, prefix + '_heatmap.png')
#
#                     # 保存误差热力图 + 同名 .npy 文件
#                     save_error_heatmap(flow_pred, flow_gt, error_heatmap_path)
#                     print(f"已生成误差热力图及误差矩阵: {error_heatmap_path} / .npy")
#                 else:
#                     print(f"预测光流与真实光流尺寸不匹配 ({flow_pred.shape} vs {flow_gt.shape})，跳过误差热力图。")
#             else:
#                 print(f"未找到真实光流文件 {flo_gt_path}，跳过误差热力图。")
#                 end_time = time.time()  # 结束计时
#                 elapsed = end_time - start_time
#                 total_time += elapsed
#                 count += 1
#                 print(f"该对图像处理耗时: {elapsed:.3f} 秒")
#
#             if count > 0:
#                 print(f"平均每对图像处理耗时: {total_time / count:.3f} 秒")
#             else:
#                 print("没有处理任何图像对。")
if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='checkpoints/250000_raft-river.pth', help="restore checkpoint")  # 模型权重路径250000_raft-river.pth
    parser.add_argument('--path',default='E:/test_data/test_river/UAV/test_V',help="dataset for evaluation")  # 数据集路径 'Tracer', 'Turbulent'
    parser.add_argument('--small', action='store_true', help='use small model')  # 是否使用小模型
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')  # 是否使用混合精度
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficient correlation implementation')  # 是否使用高效相关性实现
    parser.add_argument('--output_path',default='E:/test_data/test_river/UAV/3/RAFT', help="#COVNEXT;output path to save optical flow images")  # 输出路径
    parser.add_argument('--use_basic_layer', action='store_true', default=False, help='Whether to use the basic layer')
    args = parser.parse_args()  # 解析命令行参数

   # demo(args)  # 调用单文件：主函数
    # 获取 path 下所有子文件夹
    subfolders = [f for f in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, f))]

    if not subfolders:
        # 如果没有子文件夹，直接运行一次 demo
        print(f"📂 输入路径下无子文件夹，直接运行单次处理: {args.path}")
        demo(args)
    else:
        print(f"🔁 发现多个子文件夹，依次处理: {subfolders}")
        for sub in subfolders:
            # 构造每个子文件夹对应的输入输出路径
            sub_input_path = os.path.join(args.path, sub)
            sub_output_path = os.path.join(args.output_path, sub)

            # 创建新的 args 实例，复制原参数，但修改 path 和 output_path
            sub_args = argparse.Namespace(
                model=args.model,
                path=sub_input_path,
                small=args.small,
                mixed_precision=args.mixed_precision,
                alternate_corr=args.alternate_corr,
                output_path=sub_output_path,
                use_basic_layer=args.use_basic_layer
            )

            print(f"\n🚀 正在处理：{sub_input_path}")
            demo(sub_args)

