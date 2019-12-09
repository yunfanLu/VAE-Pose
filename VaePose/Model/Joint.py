# -*- encoding: utf-8 -*-
# @Time : 2019/3/7 11:18

import torch.nn as nn
import torch


class JointEncoder(nn.Module):

    def __init__(self, in_dim, z_dim):
        """
        关键点的输入，2d 或者 2d 的，只是 in_dim 的形式不要。
        :param in_dim: 手势关键点的维度，2维或者3维。
        :param z_dim: (3, N) or (2, N)
        """
        super(JointEncoder, self).__init__()
        in_dim = torch.IntTensor(in_dim)
        self.in_size = in_dim.prod()  # float, 返回输入张量input 所有元素的积。
        self.linear_layers = nn.Sequential(
            nn.Linear(self.in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * z_dim))

    def forward(self, x):
        return self.linear_layers(x.view(-1, self.in_size))


class JointDecoder(nn.Module):

    def __init__(self, z_dim, out_dim):
        """
        将隐空间的向量解码到手势关键点的形式，解码成二维或者三维的。
        :param z_dim:
        :param out_dim:
        """
        super(JointDecoder, self).__init__()
        self.out_dim = torch.IntTensor(out_dim)
        self.out_size = self.out_dim.prod()
        self.linear_layers = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_size.data.numpy()))

    def forward(self, x):
        out = self.linear_layers(x)
        return out.view(-1, self.out_dim[0], self.out_dim[1])
