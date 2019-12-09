# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 22:40
# @Author  : yunfan


from torch import nn

import torch

features_num = 128

m = nn.Linear(128, 128)
input = torch.randn(1, 128)
output = m(input)
print(output.shape)

mm = nn.Sequential(
    nn.Linear(features_num, features_num),
    nn.ReLU(),
    nn.BatchNorm1d(features_num),
    nn.Linear(features_num, features_num),
    nn.ReLU(),
    nn.BatchNorm1d(features_num),
    nn.Linear(features_num, features_num)
)
oo = mm(input)
print(oo.shape)