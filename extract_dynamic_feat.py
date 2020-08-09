# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import numpy as np
from models.i3d import I3D
import cv2
from PIL import Image



def read_label(label_path):
    fr = open(label_path, 'r')
    labels = []
    for i, line in enumerate(fr):
        if i == 0:
            continue
        line = line.strip().split()
        labels.append([line[0], float(line[1]),
                       float(line[2]), float(line[3])])
    return labels


class Dataset1(Dataset):
    def __init__(self, data_path, label_path, transform, sz=(224, 224), fps=5):
        super(Dataset1, self).__init__()
        self.data_path = data_path
        self.fps = fps
        self.transform = transform
        self.sz = sz
        self.label = read_label(label_path)
        self.frames_per_clip = 16

    def read_video(self, path):
        vid = cv2.VideoCapture(path)
        frames_num = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        initial_fps = vid.get(cv2.CAP_PROP_FPS)

        # 处理直接用fps/initial_fps的截断误差问题
        stride = int(frames_num / int(frames_num * self.fps / initial_fps))
        # (frames_num+stride-1) // stride 边界情况
        frames = torch.zeros(((frames_num+stride-1) // stride, 3, self.sz[0], self.sz[1]),
                             dtype=torch.float)
        w, h = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print(w, h)
        #exit()
        cnt = 1
        while True:
            ok, frame = vid.read()
            if not ok:
                break
            if cnt % stride == 0:
                frame = cv2.resize(frame, (int(w/h*224+0.5), 224))
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                frame = self.transform(frame)
                frames[cnt // stride - 1, :, :, :] = frame
            cnt += 1
        vid.release()
        return frames

    def __getitem__(self, index):
        video_path = self.data_path + self.label[index][0] + '.mp4'
        frames = self.read_video(video_path)
        cnt = len(frames) // self.frames_per_clip
        frames = frames[:self.frames_per_clip * cnt]
        # [N, C, H, W] => [N, T, C, H, W]
        frames = frames.view(-1, self.frames_per_clip, 3, self.sz[0], self.sz[1])
        # [N, T, C, H, W] => [N, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)
        return frames, self.label[index]
        # return frames, self.label[index][3]

    def __len__(self):
        return len(self.label)


def extract_i3d_feat(data_path, label_path, model_path, save_dir):
    transform = transforms.Compose([
        transforms.CenterCrop((224, 300)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dset = Dataset1(data_path, label_path, transform, sz=(224, 224), fps=2)
    dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    model = I3D(400, modality='rgb')
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    batch_size = 40

    with torch.no_grad(), tqdm(total=len(dset)) as pbar:
        for i, (data, label) in enumerate(dloader):
            data = data.cuda()
            data = data.squeeze(0)
            feat = []
            for j in range((len(data) + batch_size-1) // batch_size):
                end = min(j * batch_size + batch_size, len(data))
                feat.append(model(data[j * batch_size:end], stop='avg').squeeze().cpu().numpy())
            feat = np.vstack(feat)
            np.save(save_dir+label[0][0]+'.npy', feat)
            pbar.update(1)


if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_path = '/mnt/sdd/lingan/datasets/RhythmicGymnastics/final_datas/'
    label_path = '/mnt/sdd/lingan/datasets/RhythmicGymnastics/final_labels.txt'

    model_path = './pretrained_models/i3d_rgb.pth'
    save_dir = './data/i3d_avg_fps2_clip16a/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    extract_i3d_feat(data_path, label_path, model_path, save_dir=save_dir)

