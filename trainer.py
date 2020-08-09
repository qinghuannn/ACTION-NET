# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from scipy.stats import spearmanr

from models.action_net import ACTION_NET
from dataset import Dset
from utils import weight_init, AverageMeter


class Trainer(object):
    def __init__(self, conf, action_type='Ball', use_cuda=True):
        self.conf = conf
        self.train_loader, self.test_loader = self.get_data_loader(conf)

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.load_model(conf).to(self.device)
        self.optim = self.get_optim(conf, action_type)
        self.loss_fn = nn.MSELoss().cuda()
        self.best_coef = 0

        self.epochs = conf.epochs[action_type]
        tmp = [self.epochs-100, self.epochs-50]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, [int(x) for x in tmp], 0.1)

    def load_model(self, conf):
        print('Creating model...')

        model = ACTION_NET(clip_num=conf.clip_num, image_num=conf.image_num)
        flops, params = profile(model, inputs=(
            torch.zeros(1, 28, 1024), torch.zeros(1, 80, 2048))
        )
        model.apply(weight_init)

        print('params: %f, flops: %f' % (params, flops))
        return model

    def test(self):
        print('loading model weights...')
        self.model.load_state_dict(torch.load('./pretrained_model/Ball.pth'))
        self.test_epoch()

    def get_data_loader(self, conf, kind):
        train_dset = Dset(conf.vid_feat_path, conf.img_feat_path, conf.train_label_path,
                        clip_num=conf.clip_num, image_num=conf.image_num,
                        kind=kind)
        test_dset = Dset(conf.vid_feat_path, conf.img_feat_path, conf.test_label_path,
                        clip_num=conf.clip_num, image_num=conf.image_num,
                        rand_st=False, kind=kind)
        train_dloader = DataLoader(train_dset, batch_size=conf.batch_size,
                                   shuffle=True, num_workers=8)
        test_dloader = DataLoader(test_dset, batch_size=conf.batch_size,
                                  shuffle=False, num_workers=8)

        return train_dloader, test_dloader

    def get_optim(self, conf):
        regressor = nn.ModuleList([self.model.fc])
        regressor_params = list(map(id, regressor.parameters()))
        backbone_params = filter(lambda p: id(p) not in regressor_params,
                                 self.model.parameters())
        optim = torch.optim.SGD([
            {'params': regressor.parameters(), 'lr': conf.lr * 5},
            {'params': backbone_params}],
            lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)

        return optim

    def go(self):
        print('Training...')

        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            self.test_epoch()

        #torch.save(self.model.state_dict(),  './pretrained_models/V1_Ball.pth')

        return self.best_coef

    def train_epoch(self, epoch):
        losses = AverageMeter()
        preds = np.array([])
        labels = np.array([])

        self.model.train()
        self.scheduler.step(epoch)

        for i, (vid_feat, img_feat, label) in enumerate(self.train_loader):
            vid_feat = vid_feat.to(self.device)
            img_feat = img_feat.to(self.device)
            label = label.float().to(self.device)

            pred = self.model(vid_feat, img_feat)
            loss = self.loss_fn(pred, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.update(loss.item(), pred.size(0))
            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

        coef, _ = spearmanr(preds * 25, labels * 25)

        print('Epoch: [%d/%d]\tLoss %.4f\tCoef %.3f' % (
            epoch, self.epochs, losses.avg, coef))
        return losses.avg, coef

    def test_epoch(self):
        losses = AverageMeter()
        preds = np.array([])
        labels = np.array([])

        self.model.eval()

        with torch.no_grad():
            for i, (vid_feat, img_feat, label) in enumerate(self.test_loader):
                vid_feat = vid_feat.to(self.device)
                img_feat = img_feat.to(self.device)
                label = label.float().to(self.device)
                pred = self.model(vid_feat, img_feat)
                loss = self.loss_fn(pred, label)

                losses.update(loss.item(), pred.size(0))
                if len(preds) == 0:
                    preds = pred.cpu().detach().numpy()
                    labels = label.cpu().detach().numpy()
                else:
                    preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                    labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

        coef, _ = spearmanr(preds * 25, labels * 25)
        print('Test: Loss %.4f\tCoef %.3f' % (losses.avg, coef))

        return losses.avg, coef