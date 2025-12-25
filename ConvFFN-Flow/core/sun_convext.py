import torch
import torch.nn as nn
import time
import sys
sys.path.append('core')
from timm.layers  import trunc_normal_, DropPath
import torch.nn.functional as F
import datasets
import numpy as np
import math
from ptflops import get_model_complexity_info
from torchvision.ops import DeformConv2d

class DeformConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Offset 分支：不变
        self.offset_conv = nn.Conv2d(input_dim, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)

        # Depthwise deformable conv：group=input_dim
        self.deform_conv_dw = DeformConv2d(input_dim, input_dim, kernel_size=kernel_size,
                                           stride=stride, padding=padding, groups=input_dim, bias=False)

        # Pointwise conv：通道融合
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        offset = self.offset_conv(x)
       # print(f"Offset shape: {offset.shape}")
        x = self.deform_conv_dw(x, offset)  # Depthwise deformable conv
        x = self.pointwise(x)  # 通道融合
        return x
# class DeformConvBlock(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size=7, stride=1, padding=3):
#         super().__init__()
#         # offset预测卷积，stride和deform conv保持一致，保证尺寸匹配
#         self.offset_conv = nn.Conv2d(input_dim, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
#         # 变形卷积，不支持groups>1
#         self.deform_conv = DeformConv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding,
#                                         bias=False)
#
#     def forward(self, x):
#        # print(f"Input shape: {x.shape}")
#         offset = self.offset_conv(x)
#         print(f"Offset shape: {offset.shape}")
#         x = self.deform_conv(x, offset)
#        # print(f"Output shape: {x.shape}")
#         return x

'''该脚本是为了实现resnet残差模块替换成convnext形式，并且建立的下采样采用resnext的多分组'''
class PositionEmbeddingSine(nn.Module):
    """
    这个是用于vit的特征编码操作
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class LayerNorm(nn.Module):
    """
    输入;(batch_size, height, width, channels)----》channels_last
    输出：(batch_size, height, width, channels)
    输入;(batch_size, channels, height, width)----》channels_last
    输出：(batch_size, channels, height, width)->channels_first
    计算输入的均值和方差来进行归一化，通常用于处理每个通道独立的情况
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    它是一种归一化操作(全局）计算 L2 范数归一化
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class convnextv2_Block(nn.Module):
    """ ConvNeXtV2 Block with residual downsampling support
    Args:
        input_dim (int): Number of input channels
        dim (int): Number of output channels
        stride (int): Spatial downsampling factor
    """

    def __init__(self, input_dim, dim, drop_path=0., stride=1):
        super().__init__()
        # 主路径
        self.dwconv = nn.Conv2d(input_dim, dim, kernel_size=7, padding=3, groups=input_dim,stride=stride )#原始
        #self.dwconv = DeformConvBlock(input_dim, dim, kernel_size=7, stride=stride, padding=3)

        # 残差路径下采样模块
        self.downsample = None
        if stride != 1 or input_dim != dim:
            downsample_layers = []
            # 通道调整
            if input_dim != dim:
                downsample_layers.append(nn.Conv2d(input_dim, dim, kernel_size=1, stride=1))
            # 空间下采样
            if stride != 1:
                downsample_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            # 标准化层
            downsample_layers.append(LayerNorm(dim, data_format="channels_first"))
            self.downsample = nn.Sequential(*downsample_layers)

        # 主路径处理
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim,2* dim)
        self.act = nn.GELU()
        self.grn = GRN(2 * dim)#归一化
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        # 主路径处理
        x = self.dwconv(x)
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)  # [B,C,H,W] => [B,H,W,C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B,H,W,C] => [B,C,H,W]

        # 残差路径处理
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 维度断言检查
        assert x.shape == identity.shape, \
            f"Dimension mismatch: Main {x.shape} vs Residual {identity.shape}"

        return self.act(x + identity)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=256, dropout=0.0,alpha = 0.05 ):
        super(BasicEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #self.conv1 = DeformConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = LayerNorm(64, eps=1e-6, data_format="channels_first")
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.GELU()
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)

        self.conv2 = nn.Conv2d(256, output_dim, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        self.alpha=alpha
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # 更通用，对 GELU 友好,初始化

            elif isinstance(m, (nn.BatchNorm2d,  LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = convnextv2_Block(self.in_planes, dim,stride=stride)
        layer2 = convnextv2_Block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)
    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        # x = self.relu1(x)##分辨率减半(B, C, H, W)-->(B, C, H/2, W/2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        pos_enc = PositionEmbeddingSine(num_pos_feats=x.shape[1] // 2)
        position = pos_enc(x)

        x = x + self.alpha *position

        if self.dropout is not None and self.training:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


def test_training_time():
    # 配置参数
    batch_size = 8
    input_size = (432, 768)  # 原测试用例尺寸
    num_warmup = 20  # 预热迭代次数
    num_repeats = 50  # 正式测量次数

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")

    # 模型初始化
    model = BasicEncoder().to(device)
    model.train()  # 确保训练模式

    # 创建示例数据
    dummy_input = torch.randn(batch_size, 3, *input_size).to(device)
    target = torch.randn(batch_size, 256, 54, 96).to(device)  # 根据输出形状调整

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 预热阶段
    print("开始预热...")
    for _ in range(num_warmup):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()  # 等待所有CUDA操作完成

    # 正式时间测量
    print("开始正式计时...")
    total_time = 0.0

    # 使用性能分析器
    # 修改后的性能分析部分
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_repeats),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/basic_encoder'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for _ in range(num_repeats):
            start_time = time.time()

            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()  # 精确计时关键步骤
            end_time = time.time()

            if device.type == 'cuda':
                prof.step()  # 更新性能分析器
                total_time += (end_time - start_time) * 1000  # 转换为毫秒

    # 结果分析
    avg_time = total_time / num_repeats
    print(f"\n=== 测试结果 ===")
    print(f"单批次平均训练时间: {avg_time:.2f} ms")
    print(f"每秒可处理批次: {1000 / avg_time:.1f} batch/s")

    # 输出性能分析摘要
    print("\n=== 性能分析摘要 ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total",
        row_limit=10
    ))

    # 显存使用情况
    if device.type == 'cuda':
        print(f"\n最大显存占用: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

import matplotlib.pyplot as plt

def visualize_feature_maps(output, num_images=8, num_channels=5):
    """ 可视化特征图的前 num_channels 个通道 """
    batch_size, num_channels, height, width = output.shape

    # 选取前 num_images 个样本
    for i in range(min(num_images, batch_size)):
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
        for j in range(min(num_channels, num_channels)):  # 选择前 num_channels 个通道
            feature_map = output[i, j, :, :].cpu().detach().numpy()  # 取出特定通道
            axes[j].imshow(feature_map, cmap='viridis')  # 使用 'viridis' 色图
            axes[j].axis('off')  # 不显示坐标轴

        plt.title(f"Sample {i + 1} - Feature Maps")
        plt.show()

import argparse
# 测试用例
if __name__ == "__main__":

    # 设置随机种子
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='River_data', help="determines which dataset to use for training")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, nargs='+', default=[400, 400])  # 行列432,768
    args = parser.parse_args()

    # 创建模型
    model = BasicEncoder()
    # 将模型移动到 GPU 上
    model = model.cuda()
    # 获取训练数据加载器
    train_loader = datasets.fetch_dataloader(args)

    # 迭代 train_loader 获取批次数据
    for i_batch, data_blob in enumerate(train_loader):
        # 假设 data 是输入特征, target 是标签

        # 将数据迁移到GPU（如果使用GPU的话）
        data, target = data.cuda(), target.cuda()  # 如果你有GPU，可以使用 .cuda()
        # 将图像和光流数据转移到 GPU 上
        image1, image2, flow, valid = [x.cuda() for x in data_blob]
        # 获取 image1 和 image2 的特征
        feature1 = model(image1)  # 对第一张图像进行前向传播，获取特征
        feature2 = model(image2)  # 对第二张图像进行前向传播，获取特征

        # 可视化前 5 个通道的特征图
        visualize_feature_maps(feature1, num_images=2, num_channels=5)

    # # 计算总参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"模型总参数量: {total_params:,}")  # 格式化输出
    # # 更直观的显示方式
    # print(f"\n参数规模: {total_params / 1e6:.2f} 百万参数")  # 转换为百万单位
    # macs, params = get_model_complexity_info(model, (3, 400, 768), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    ##########FLOPs（MACs）: 22.69 GMac
    # 参数量: 1.05 M

# #
#     print(f'FLOPs（MACs）: {macs}')
#     print(f'参数量: {params}')


    # 运行测试（建议在Jupyter notebook外单独运行）
   # test_training_time()
