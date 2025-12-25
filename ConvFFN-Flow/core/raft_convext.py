import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
# from extractor import BasicEncoder, SmallEncoder
from sun_convext import BasicEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
import newcnn
##############################这个才是我的实验的模型
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        #
        self.fnet = BasicEncoder(output_dim=256, dropout=args.dropout)  # 特征维度为256，长宽根据数据加载器提供
        self.cnet = BasicEncoder(output_dim=hdim + cdim, dropout=args.dropout)  # 特征维度为256，长宽根据数据加载器提供
        # self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        # self.fnet = newcnn.ConvNeXt_1_8_Feature_3(in_chans=3, depths=[1, 3, 1], dims=[64, 128, 256],
        #                                    drop_path_rate=0., layer_scale_init_value=1e-6)  # 特征维度为256，长宽根据数据加载器提供
        # self.cnet = newcnn.ConvNeXt_1_8_Feature_3(in_chans=3, depths=[1, 3, 1], dims=[64, 128, hdim + cdim],
        #                                    drop_path_rate=0., layer_scale_init_value=1e-6)  # 特征维度为256，长宽根据数据加载器提供
        # self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout) # 特征维度为256，长宽根据数据加载器提供
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=4, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # 标准线性归一化convnext
        # 先将图像范围缩放到 [0,1]
        image1 = image1 / 255.0
        image2 = image2 / 255.0

        # 使用 ConvNeXt 推荐的标准化方式
        mean = torch.tensor([0.485, 0.456, 0.406], device=image1.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image1.device).view(1, 3, 1, 1)

        image1 = (image1 - mean) / std
        image2 = (image2 - mean) / std

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()

        # def hook_fn(grad):
        #     print("[Hook] fmap1 grad mean:", grad.abs().mean().item())
        #
        # fmap1.register_hook(hook_fn)  # x 是 layer3 的输出
        fmap2 = fmap2.float()

        # def hook_fn(grad):
        #     print("[Hook] fmap2 grad mean:", grad.abs().mean().item())
        #
        # fmap2.register_hook(hook_fn)  # x 是 layer3 的输出
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)


        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
                                                                                    #print(corr.shape)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

                # def hook_fn(grad):
                #     print("[Hook] update_block grad mean:", grad.abs().mean().item())
                #
                # fmap2.register_hook(hook_fn)  # x 是 layer3 的输出

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
