# -*- encoding: utf-8 -*-
# @Time : 2019/2/26 18:33

import torch


def videoEcoderTest():
    import VaePose.Model as vae
    bz, n_frames, z_dim, c, w, h = 32, 2, 32, 3, 256, 256
    video_encoder = vae.VideoEncoder(n_frames, z_dim, False)
    X = torch.randn(n_frames, c, w, h)
    print(video_encoder(X).shape)


if __name__ == '__main__':
    videoEcoderTest()
