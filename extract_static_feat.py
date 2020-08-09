# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
from PIL import Image
from sklearn.svm import SVR
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


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


def extract_frames(video_dir, save_dir, sample_rate=1):
    def extract_frames_from_video(video_name):
        video_path = video_dir + video_name + '.mp4'
        vid = cv2.VideoCapture(video_path)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        frames_num = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = int(frames_num / int(frames_num * sample_rate / fps))
        cnt = 0
        while True:
            ok, frame = vid.read()
            cnt += 1
            if ok is False:
                break
            if cnt % stride != 0:
                continue
            cv2.imwrite(save_dir + video_name + '_%d.jpg' % (cnt // stride), frame)
        vid.release()

    video_names = [x.split('.')[0] for x in sorted(os.listdir(video_dir))]

    executor = ThreadPoolExecutor(max_workers=8)
    all_task = [executor.submit(extract_frames_from_video, (x)) for x in video_names]
    with tqdm(total=len(video_names)) as pbar:
        for _ in as_completed(all_task):
            pbar.update(1)

    print('extract all videos done!')


class Dataset1(Dataset):
    def __init__(self, data_path, label_path, transform, sz=(224, 224)):
        self.data_path = data_path
        self.transform = transform
        self.sz = sz
        self.label = read_label(label_path)
        self.img_names = sorted(os.listdir(data_path))

    def __getitem__(self, index):
        prefix = self.label[index][0]
        len_prefix = len(prefix)
        img_names = [x for x in self.img_names if x[:len_prefix] == prefix]
        imgs = torch.zeros([len(img_names), 3, self.sz[0], self.sz[1]], dtype=torch.float)
        for i, img_name in enumerate(img_names):
            img = Image.open(self.data_path+img_name).convert('RGB')
            w, h = img.size
            img = img.resize((int(w / h * 224 + 0.5), 224), Image.BILINEAR)
            imgs[i, :, :, :] = self.transform(img)
        return imgs, self.label[index]

    def __len__(self):
        return len(self.label)


def extract_from_raw_img(data_path, label_path,  save_dir):
    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Linear(2048, 2048)
    torch.nn.init.eye_(resnet.fc.weight)
    resnet = resnet.cuda()
    resnet.eval()

    dset = Dataset1(data_path, label_path, transform)
    dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        with tqdm(total=len(dset)) as pbar:
            for data, label in dloader:
                data = data.cuda()
                feat = resnet(data[0])
                np.save(save_dir + label[0][0] + '.npy', feat.cpu().numpy())
                pbar.update(1)


class Dataset2(Dataset):
    def __init__(self, data_path, label_path, transform, sz=(224, 224)):
        self.data_path = data_path
        self.transform = transform
        self.sz = sz
        self.label = read_label(label_path)
        self.img_names = sorted(os.listdir(data_path))

    def __getitem__(self, index):
        prefix = self.label[index][0]
        len_prefix = len(prefix)
        img_names = [x for x in self.img_names if x[:len_prefix] == prefix]
        imgs = torch.zeros([len(img_names), 3, self.sz[0], self.sz[1]], dtype=torch.float)
        for i, img_name in enumerate(img_names):
            img = Image.open(self.data_path+img_name).convert('RGB')
            imgs[i, :, :, :] = self.transform(img)
        return imgs, self.label[index]

    def __len__(self):
        return len(self.label)


def extract_feat_from_cropped_img(data_path, label_path, save_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Linear(2048, 2048)
    torch.nn.init.eye_(resnet.fc.weight)
    resnet = resnet.cuda()
    resnet.eval()

    dset = Dataset2(data_path, label_path, transform)
    dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        with tqdm(total=len(dset)) as pbar:
            for data, label in dloader:
                data = data.cuda()
                feat = resnet(data[0])
                np.save(save_dir + label[0][0] + '.npy', feat.cpu().numpy())
                pbar.update(1)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-op', type=str, default='raw_image')
    parse.add_argument('-gpu', type=str, default='0')
    args = parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = '/mnt/sdd/lingan/datasets/RhythmicGymnastics/final_datas/'
    raw_image_path = './data/raw_image/'
    cropped_image_path = './data/cropped_image_fps0.5/'
    label_path = '/mnt/sdd/lingan/datasets/RhythmicGymnastics/final_labels.txt'
    if args.op == 'raw_image':
        # save_dir = './data/raw_image_feat/'
        save_dir = './data/raw_image_feat/'
        extract_from_raw_img(raw_image_path, label_path, save_dir)
    elif args.op == 'cropped_image':
        # save_dir = './data/crop_image_feat/'
        #save_dir = './data/new_crop_image_feat/'
        save_dir = './data/crop_image_feat_fps0.5/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        extract_feat_from_cropped_img(cropped_image_path, label_path, save_dir)
    elif args.op == 'frames':
        #save_dir = './data/images/'
        save_dir = './data/images_fps4/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        extract_frames(video_path, save_dir, sample_rate=4)