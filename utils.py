# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import logging
from tensorboardX import SummaryWriter


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def Logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.setLevel(level)
        logger.addHandler(handler)
        #logger.addHandler(console)

    return logger


class Visualizer(object):
    def __init__(self, logdir='./runs'):
        self.writer = SummaryWriter(logdir)

    def write_graph(self, model, dummy_input):
        self.writer.add_graph(model, (dummy_input, ))

    def write_summary(self, info_coef, info_loss, epoch):
        self.writer.add_scalars('Spearman Rank cofficient', info_coef, epoch)
        self.writer.add_scalars('Loss', info_loss, epoch)

    def write_histogram(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

    def writer_close(self):
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def score_smoothing(score, r=2, sigma=2):
    guass_filter = np.zeros([r * 2 + 1])
    for i in range(r * 2 + 1):
        guass_filter[i] = np.exp(-1 * (i - r) ** 2 / (2 * sigma) ** 2) / (
                np.sqrt(2 * np.pi) * sigma)
    for i in range(r, score.shape[1]-r):
        score[:, i] = np.dot(score[:, i - r:i + r + 1], guass_filter)
    return score
