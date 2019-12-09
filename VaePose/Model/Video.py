# -*- encoding: utf-8 -*-
# @Time : 2019/3/7 11:21

import torch.nn as nn
from . import RGBEncoder

class VideoEncoder(nn.Module):
    def __init__(self, n_frames, z_dim, hand_side_invariance):
        """
        实现对视频，多帧图片的同时编码。
        :param n_frames:
        :param z_dim:
        :param hand_side_invariance:
        TODO: BatchNormal 的正确写法
        """
        super(VideoEncoder, self).__init__()
        self.n_frames = n_frames
        self.z_dim = z_dim
        self.hand_side_invariance = hand_side_invariance
        self.frameEncoder = RGBEncoder(z_dim, hand_side_invariance)
        features_num = self.n_frames * self.z_dim * 2
        self.layers = nn.Sequential(
            nn.Linear(features_num, features_num),
            nn.ReLU(),
            # nn.BatchNorm1d(features_num),
            nn.Linear(features_num, features_num),
            nn.ReLU(),
            # nn.BatchNorm1d(features_num),
            nn.Linear(features_num, features_num))

    def forward(self, x):
        """
        N 张图片一起输入。
        :param x:(batch_size, n, 3, 320, 320)
        :return:
        """
        # TODO: 这里的是 0 还是 1
        assert x.size(0) == self.n_frames
        y = self.frameEncoder(x)
        y = y.view(1, -1)
        res = y
        y = self.layers(y)
        o = res + y
        o = o.view(self.n_frames, -1)
        return o