# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch

import conf
from trainer import Trainer


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False  # 训练集变化不大时使训练加速


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='-1')
    parser.add_argument('-action_type', type=str, default='Ball')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')

    args = parser.parse_args()

    use_gpu = False
    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        use_gpu = True

    setup_seed(0)

    trainer = Trainer(conf, args.action_type, use_gpu)

    if args.train:
        trainer.go()
    elif args.test:
        trainer.test()
