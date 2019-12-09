# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 14:14
# @Author  : yunfan
import pickle


def getRHDLabel():

    with open('RHD/training/anno_training.pickle', 'rb') as f:
        anno_all = pickle.load(f)

    return anno_all

if __name__ == '__main__':
    ann = getRHDLabel()
    print(len(ann.keys()))
    l1 = ann[1]
    print(l1.keys())