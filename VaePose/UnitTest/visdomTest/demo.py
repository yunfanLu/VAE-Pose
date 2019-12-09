# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 21:43
# @Author  : yunfan

import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))

plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots')