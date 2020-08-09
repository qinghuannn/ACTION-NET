# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Dset(Dataset):
    '''
    crop imag feat + video i3d feat
    '''
    def __init__(self, video_feat_path, image_feat_path, label_path,
                 score_type='Total_Score', clip_num=28, image_num=80, rand_st=True, kind='Ball'):
        self.data_path1 = video_feat_path
        self.data_path2 = image_feat_path
        self.clip_num = clip_num
        self.image_num = image_num
        self.rand_st = rand_st
        self.label = self.read_label(label_path, score_type, kind)

    def read_label(self, label_path, score_type, kind):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if kind == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx[score_type]])])
        return labels

    def __getitem__(self, idx):
        vid_feat = np.load(self.data_path1 + self.label[idx][0] + '.npy')
        st = (len(vid_feat) - self.clip_num) // 2
        if self.rand_st and len(vid_feat) != self.clip_num:
            st = np.random.randint(0, len(vid_feat) - self.clip_num)
        vid_feat = vid_feat[st:st + self.clip_num]
        vid_feat = torch.from_numpy(vid_feat).float()

        img_feat = np.load(self.data_path2 + self.label[idx][0] + '.npy')
        if len(img_feat) < self.image_num:
            raw_feat = np.zeros([self.image_num, img_feat.shape[1]])
            raw_feat[:len(img_feat)] = img_feat
            img_feat = raw_feat
        st = (len(img_feat) - self.image_num) // 2
        if self.rand_st and len(img_feat) != self.image_num:
            st = np.random.randint(0, len(img_feat) - self.image_num)
        img_feat = img_feat[st:st + self.image_num]
        img_feat = torch.from_numpy(img_feat).float()

        return vid_feat, img_feat, self.label[idx][1] / 25

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    vid_feat_path = './data/i3d_avg_fps5_clip16a/'
    img_feat_path = './data/crop_image_feat_fps0.5/'
    train_label_path = './data/final_train.txt'

    dset = Dset(vid_feat_path, img_feat_path, train_label_path)
    dloader = DataLoader(dset, batch_size=1, shuffle=False)

    for vid_feat, img_feat, label in dloader:
        print(vid_feat.size(), img_feat.size(), label.size())
        break
