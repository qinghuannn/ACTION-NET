# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dset(Dataset):
    '''
    crop imag feat + video i3d feat
    '''
    def __init__(self, video_feat_path, image_feat_path, label_path,
                clip_num=26, image_num=80, rand_st=True, action_type='Ball',
                score_type='Total_Score'):
        self.data_path1 = video_feat_path
        self.data_path2 = image_feat_path
        self.clip_num = clip_num
        self.image_num = image_num
        self.rand_st = rand_st
        self.label = self.read_label(label_path, score_type, action_type)

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx[score_type]])])
        return labels

    def __getitem__(self, idx):
        dynamic_feat = np.load(self.data_path1 + self.label[idx][0] + '.npy')
        st = (len(dynamic_feat) - self.clip_num) // 2
        if self.rand_st and len(dynamic_feat) != self.clip_num:
            st = np.random.randint(0, len(dynamic_feat) - self.clip_num)
        dynamic_feat = dynamic_feat[st:st + self.clip_num]
        dynamic_feat = torch.from_numpy(dynamic_feat).float()

        static_feat = np.load(self.data_path2 + self.label[idx][0] + '.npy')
        if len(static_feat) < self.image_num:
            raw_feat = np.zeros([self.image_num, static_feat.shape[1]])
            raw_feat[:len(static_feat)] = static_feat
            static_feat = raw_feat
        st = (len(static_feat) - self.image_num) // 2
        if self.rand_st and len(static_feat) != self.image_num:
            st = np.random.randint(0, len(static_feat) - self.image_num)
        static_feat = static_feat[st:st + self.image_num]
        static_feat = torch.from_numpy(static_feat).float()

        return dynamic_feat, static_feat, self.label[idx][1] / 25

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':

    dynamic_feat_path = './data/dynamic_feat/'
    static_feat_path = './data/static_feat/'
    train_label_path = './data/train.txt'
    test_label_path = './data/test.txt'
    clip_num = 26
    image_num = 80

    dset = Dset(dynamic_feat_path, static_feat_path, test_label_path,
                clip_num=clip_num, image_num=image_num, rand_st=False)
    dloader = DataLoader(dset, batch_size=1, shuffle=False)

    for dynamic_feat, static_feat, label in dloader:
        print(dynamic_feat.size(), static_feat.size(), label.size())
        break
