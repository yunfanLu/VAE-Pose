# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 17:59
# @Author  : yunfan

import yaml
from easydict import EasyDict as edict

def get_config(cfg_yaml_path):
    with open(cfg_yaml_path) as f:
        cfg = edict(yaml.load(f))
    return cfg