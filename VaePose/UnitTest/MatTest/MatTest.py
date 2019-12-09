# -*- encoding: utf-8 -*-
# @Time : 2019/2/27 13:08

import scipy.io
import os

labels_path = "D:/Dataset/Hand Pose 2d to 3d/STB/labels"
mat_files = os.listdir(labels_path)
lab = {}
for mat in mat_files:
    path = os.path.join(labels_path, mat)
    v = scipy.io.loadmat(path)
    # print(v.keys())
    v = v['handPara']
    # print(f'{mat}\t\t\t{type(v)}\t{v.shape}') # (3, 21, 1500)
    # print(v[:,:,:1])
    # break
    lab[mat] = v
print(lab.keys())