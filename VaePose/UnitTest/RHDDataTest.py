# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 22:40
# @Author  : yunfan

import numpy as np
from torch.utils.data import DataLoader
from VaePose.data.RHDDataset import RHDDataset
from VaePose.utils.View import view_origin_img, view_hand_2d, view_mask
from VaePose.config.rhd import RHDConfig as cfg
from VaePose.data.Augment import augment_data



def viewTheData():
    RHDPath = '/Volumes/YF-Code-256/Data/RHD/RHD/'
    RHD_val_ds = RHDDataset(RHDPath, 'evaluation')
    RHDLoader = DataLoader(RHD_val_ds, batch_size=1, shuffle=False)

    for sample in RHDLoader:
        rgb = sample['rgb']
        abs_d2d = sample['abs_2d']
        rel_2d = sample['rel_2d']
        rel_3d = sample['rel_3d']
        scale = sample['scale']
        depth = sample['depth']
        mask = sample['mask']
        break


    print(rgb.shape, type(rgb))
    print(abs_d2d.shape, type(abs_d2d))
    print(rel_2d.shape, type(rel_2d))
    print(rel_3d.shape, type(rel_3d))
    print(depth.shape, type(depth))
    print(scale)

    # im = view_origin_img(rgb, show=False, save=False)
    # print(d2d.shape, type(d2d))
    # view_hand_2d(d2d, cfg.colorlist_pred, im, show=False, save=False)
    # view_mask(mask, show=False, save=False)

    # nrgb, abs_n2d, aug_n2d, n3d = augment_data(prng, rgb.numpy()[0], d2d.numpy()[0], d3d.numpy()[0],
    #                                            crop_size=(256,256), rotate_aug=True, flip_aug=True)
    # print(nrgb.shape, type(nrgb))
    # print(abs_n2d.shape, type(abs_n2d))
    # print(aug_n2d.shape, type(aug_n2d))
    # print(n3d.shape, type(n3d))
    # im = view_origin_img(nrgb, show=True, save=False)
    # view_hand_2d(abs_n2d, cfg.colorlist_gt, im, save=False)
    # view_hand_2d(aug_n2d[0] + np.array([160, 160]), cfg.colorlist_gt, save=False)
    # view_mask(mask, save=False)


if __name__ == '__main__':
    prng = np.random.RandomState(123)
    viewTheData()