# -*- encoding: utf-8 -*-
# @Time : 2019/2/20 18:31

from torch.autograd import Variable
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, z_dim, encoder, decoder):
        """
        the VAE model,
        :param z_dim: int
        :param encoder: nn.Model
        :param decoder: nn.Model
        """
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        """
        Split to mu and logvar，正态分布的协方差矩阵是对角矩阵，所以这里编码之后，前面
        z_dim 个元素组成的向量是均值，后面 z_dim 个元素组成的向量是方差的对角线。
        :param x:
        :param hand_side:
        :return:
        """
        h_i = self.encoder(x)
        return h_i[:, self.z_dim:], h_i[:, :self.z_dim]

    def reparameterize(self, mu, logvar):
        # TODO, 2019年03月04日 在训练的时候，参数一个符合这个正太分布的随机数，这里最好改成多个。
        # TODO，形成一个[(p, mu)...]的序列对，前面表示的是这个拿到这个 mu 的概率，也就是正态分布在
        # TODO，这一点的值，后者是这一点的坐标是一个 z_dim 维的向量。
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            mu

    def forward(self, x, vae_decoder=None, hand_size=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec = vae_decoder if vae_decoder else self.decoder
        return dec(z), mu, logvar
