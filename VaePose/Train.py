# -*- encoding: utf-8 -*-
# @Time : 2019/2/20 16:00

# import pudb
# pudb.set_trace()

import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from VaePose.data.RHDDataset import getRHDTrainValDataset
from VaePose.config.rhd import RHDConfig as rhgcfg
from VaePose.Model import RGBDecoder, VAE, JointEncoder, JointDecoder, RGBEncoder
from VaePose.utils.tools import create_exp_folder
from VaePose.utils.tools import is_windows
from VaePose.Loss.loss import mse, kl_div, loss_pck_fn


def directory_setup(output_folder):
    """
    create folder exp_XXX/ in output_folder and make source_files folder.
    :param output_folder: ./vae_rhd_output
    :return: exp_XXX folder path
    """
    exp_folder = create_exp_folder(output_folder)
    return exp_folder


def vae_forward_pass(input, target, model, losses, hand_side=None, scale=None,
                     weight=None, bp=True):
    batch_size = input.size(0)
    scale = 1 if scale is None else scale
    print(f"input.shape: {input.shape}") # [64, 3, 256, 256]
    recon_batch, mu, logvar = model(input) # [64, 3, 128, 128]
    mse_loss = mse(recon_batch, target, weight)
    kl_loss = kl_div(mu, logvar, input.size(-1), weights=weight, n_joints=rhgcfg.n_joints)
    if bp:
        loss = kl_loss * rhgcfg.kl_term_reg + mse_loss
        loss.backward()
    losses[0] += mse_loss.item() * batch_size
    losses[1] += kl_loss.item() * batch_size
    return kl_loss, mse_loss, recon_batch, mu, logvar


def print_loss(epoch, batch_idx, model_str, losses):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Epoch({epoch}): Batch:{batch_idx}, Time:{now}")
    print(f"\t{model_str[0]}\t{model_str[1]}\t{model_str[2]}")
    for i in range(3):
        ms = model_str[i]
        print(f"{ms}\t{losses[ms][0]}\t{losses[ms][1]}\t{losses[ms][2]}")


def train_rhd(epoch, vaes, ds_loader, optimizer, bp=True):
    vae_rgb_2_rgb, vae_rgb_2_2d, vae_rgb_2_3d, vae_2d_2_rgb, vae_2d_2_2d, vae_2d_2_3d, vae_3d_2_rgb, vae_3d_2_2d, vae_3d_2_3d = vaes
    model_str = ['rgb', '2d', '3d']
    models = {}
    losses = {}
    for m1 in model_str:
        losses[m1], models[m1] = {}, {}
        for m2 in model_str:
            losses[m1][m2] = [0, 0]  # 0 是 MSE Loss，1 是 KL Loss
            vae_str = f'vae_{m1}_2_{m2}'
            eval(vae_str).train()
            models[m1][m2] = eval(vae_str)
    pck_loss = {'pck_10': 0, 'pck_15': 0, 'pck_20': 0}

    for batch_idx, sample in enumerate(ds_loader):
        rgb = sample['rgb'].cuda()
        d2d = sample['rel_2d'].cuda()
        d3d = sample['rel_3d'].cuda()
        data = [rgb, d2d, d3d]

        if rhgcfg.dataset_weighting and ('weight' in sample.keys()):
            weight = sample['weight'].cuda()
        else:
            weight = None
        if rhgcfg.hand_side_invariance and ('hand_side' in sample.keys()):
            hand_side = sample['hand_side'].cuda()
        else:
            hand_side = None
        if rhgcfg.scale_invariance and ('scale' in sample.keys()):
            scale = sample['scale'].cuda()
        else:
            scale = Variable(torch.ones(1)).cuda()

        for i in range(3):
            for j in range(3):
                print(f"{model_str[i]} -> {model_str[j]}")
                mse_loss, kl_loss, recon_batch, mu, logvar = vae_forward_pass(
                    input=data[i], target=data[j], model=models[model_str[i]][model_str[j]],
                    losses=losses[model_str[i]][model_str[j]], hand_side=hand_side, scale=scale,
                    weight=weight, bp=bp)
                if bp == True:
                    optimizer.step()
                    optimizer.zero_grad()
                if model_str[i] == 'rgb' and model_str[j] == '3d':
                    pck_loss['pck_10'] += loss_pck_fn(recon_batch.data, d3d, 0.010)
                    pck_loss['pck_15'] += loss_pck_fn(recon_batch.data, d3d, 0.015)
                    pck_loss['pck_20'] += loss_pck_fn(recon_batch.data, d3d, 0.020)

        if batch_idx % rhgcfg.interval == 0:
            print_loss(epoch, batch_idx, model_str, losses)
    print_loss(epoch, batch_idx, model_str, losses)


def eval_rhd(vaes, eval_ds_loader):
    print(f"{'='*10} TEST {'='*10}")
    train_rhd(epoch='TEST', vaes=vaes, ds_loader=eval_ds_loader, optimizer=None, bp=False)
    print(f"{'='*10}======{'='*10}")


def main():
    # 0. set seeds
    torch.manual_seed(rhgcfg.seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rhgcfg.seed_val)
    np.random.seed(rhgcfg.seed_val)

    # 2. Directory and Logger setup
    exp_folder = directory_setup(rhgcfg.output_folder)
    rhgcfg.exp_folder = exp_folder

    # 3. Load Dataset, Per-processing
    print(f"Joint Dimension: {rhgcfg.dim_joints}")
    train_ds, validation_ds = getRHDTrainValDataset(RHDRoot)
    train_ds_loader = DataLoader(train_ds, batch_size=rhgcfg.bz, shuffle=True)
    eval_ds_loader = DataLoader(validation_ds, batch_size=rhgcfg.bz, shuffle=False)

    # 4. Model [RGB, 2D, 3D] 的顺序来。
    ## 4.1 RGB2RGB Model
    rgb_encoder = RGBEncoder(z_dim=rhgcfg.z_dim, hand_side_invariance=rhgcfg.hand_side_invariance)
    joint_2d_encoder = JointEncoder(in_dim=[rhgcfg.n_joints, 2], z_dim=rhgcfg.z_dim)
    joint_3d_encoder = JointEncoder(in_dim=[rhgcfg.n_joints, 3], z_dim=rhgcfg.z_dim)
    rgb_decoder = RGBDecoder(z_dim=rhgcfg.z_dim)  # z_dim to image(3,128,128)
    joint_2d_decoder = JointDecoder(z_dim=rhgcfg.z_dim, out_dim=[rhgcfg.n_joints, 2])
    joint_3d_decoder = JointDecoder(z_dim=rhgcfg.z_dim, out_dim=[rhgcfg.n_joints, 3])
    ## 4.2 Structure the encoder-decoder pair
    vae_rgb_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=rgb_decoder)
    vae_rgb_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=joint_2d_decoder)
    vae_rgb_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=joint_3d_decoder)
    vae_2d_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=rgb_decoder)
    vae_2d_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=joint_2d_decoder)
    vae_2d_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=joint_3d_decoder)
    vae_3d_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=rgb_decoder)
    vae_3d_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=joint_2d_decoder)
    vae_3d_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=joint_3d_decoder)
    vaes = [vae_rgb_2_rgb, vae_rgb_2_2d, vae_rgb_2_3d,
            vae_2d_2_rgb, vae_2d_2_2d, vae_2d_2_3d,
            vae_3d_2_rgb, vae_3d_2_2d, vae_3d_2_3d]

    vaes_en_de_pair = [rgb_encoder, joint_2d_encoder, joint_3d_encoder, rgb_decoder, joint_2d_decoder,
                       joint_3d_decoder]
    vaes_parameters = []
    ## 5 Drive in GPU
    for vae in vaes_en_de_pair:
        if torch.cuda.is_available():
            vae.cuda()
        vaes_parameters.append({"params": vae.parameters()})
        print(vae)
    ## 5. Set optimizer
    optimizer = torch.optim.Adam(vaes_parameters, lr=rhgcfg.lr)

    ## 6. Training
    for epoch in range(1, rhgcfg.Epochs + 1):
        train_rhd(epoch, vaes, train_ds_loader, optimizer)
        # test_rhd(vae_3d23d, vae_rgb2rgb, eval_ds_loader)
        eval_rhd(vaes, eval_ds_loader)


if __name__ == '__main__':
    if is_windows() is False:
        RHDRoot = './RHD'
    else:
        RHDRoot = 'E:\\Data\\RHD\\RHD\\'
    main()
