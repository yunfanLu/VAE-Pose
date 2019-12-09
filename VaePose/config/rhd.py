# -*- encoding: utf-8 -*-
# @Time : 2019/2/20 16:02

class RHDConfig:
    Epochs = 800
    lr = 1e-4
    bz = 128
    works = 16
    n_joints = 21
    z_dim = 32

    dim_joints = 3
    both_hands = True
    # TODO: 这里原本是 False
    hand_side_invariance = False # 左右手不变性
    scale_invariance = True # 尺度不变性
    rotate_aug = True # 是否要旋转

    filp_aug = True
    cropped_img = True

    dataset_weighting = False
    crop_size = (256, 256)

    start_epoch = 0

    # Noise
    joint_dropout_prod = 0.0
    save_frequency = 50
    gassian_noise_std = 2.5
    kl_term_reg = 1
    kl_term_inc = 0.01

    test_set_frac = 0.01
    synth_shift_factor = 0.5
    log_interval = 50

    # seeds
    seed_val = 1

    #Log
    interval = 10

    # output folder
    output_folder = './output/vae_rhd'

    exp_folder = None

