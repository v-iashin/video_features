'''
    Reference: https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868
    Modified by Vladimir Iashin for github.com/v-iashin/video_features
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.raft.raft_src.update import BasicUpdateBlock, SmallUpdateBlock
from models.raft.raft_src.extractor import BasicEncoder, SmallEncoder
from models.raft.raft_src.corr import CorrBlock, AlternateCorrBlock
from models.raft.raft_src.utils.utils import bilinear_sampler, coords_grid, upflow8

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
    def __init__(self, model_is_small):
        super(RAFT, self).__init__()
        self.dropout = 0
        self.alternate_corr = False
        self.model_is_small = model_is_small
        self.mixed_precision = False

        if self.model_is_small:
            self.corr_levels = 4
            self.corr_radius = 3
            self.hidden_dim = 96
            self.context_dim = 64
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        else:
            self.corr_levels = 4
            self.corr_radius = 4
            self.hidden_dim = 128
            self.context_dim = 128
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        # feature network, context network, and update block
        if self.model_is_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=self.cnet_out_dim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=self.cnet_out_dim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=self.mixed_precision):
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
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

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
