# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch

import conf
from trainer import Trainer


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--action_type', type=str, default='Ball')
    parser.add_argument('-test', action='store_true')

    args = parser.parse_args()

    use_gpu = False
    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        use_gpu = True

    setup_seed(0)

    trainer = Trainer(conf, args.action_type, use_gpu)

    if not args.test:
        trainer.go()
    else:
        trainer.test()
