# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def extract_frames(video_dir, save_dir, sample_rate=1):
    def extract_frames_from_video(video_name):
        if not osp.exists(osp.join(save_dir, video_name)):
            os.mkdir(osp.join(save_dir, video_name))

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
            cv2.imwrite(osp.join(osp.join(save_dir, video_name), '%03d.jpg' % (cnt // stride)), frame)
        vid.release()

    video_names = [x.split('.')[0] for x in sorted(os.listdir(video_dir))]

    executor = ThreadPoolExecutor(max_workers=8)
    all_task = [executor.submit(extract_frames_from_video, (x)) for x in video_names]
    with tqdm(total=len(video_names)) as pbar:
        for _ in as_completed(all_task):
            pbar.update(1)

    print('Done!')


class Dset(Dataset):
    def __init__(self, data_path, transform, sz=(224, 224)):
        self.data_path = data_path
        self.transform = transform
        self.sz = sz
        self.videos = sorted(os.listdir(data_path))

    def __getitem__(self, index):
        video_name = self.videos[index]
        image_names = sorted(os.listdir(osp.join(self.data_path, video_name)))
        imgs = torch.zeros([len(image_names), 3, self.sz[0], self.sz[1]], dtype=torch.float)
        for i, img_name in enumerate(image_names):
            img = Image.open(osp.join(osp.join(self.data_path, video_name), img_name)).convert('RGB')
            imgs[i, :, :, :] = self.transform(img)
        return imgs, video_name

    def __len__(self):
        return len(self.videos)



def extract_feat_from_cropped_img(data_path, save_dir):
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

    dset = Dset(data_path, transform)
    dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        with tqdm(total=len(dset)) as pbar:
            for data, video_name in dloader:
                data = data.cuda()
                feat = resnet(data[0])
                np.save(save_dir + video_name[0] + '.npy', feat.cpu().numpy())
                pbar.update(1)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--op', type=str, default='raw_image')
    parse.add_argument('--gpu', type=str, default='0')
    args = parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = '/home/xxx/datasets/RG_public/videos/'
    cropped_image_path = './data/cropped_frames/'

    if args.op == 'extract_feat':
        save_dir = './data/static_feat/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        extract_feat_from_cropped_img(cropped_image_path, save_dir)

    elif args.op == 'extract_frame':
        save_dir = './data/frames/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        extract_frames(video_path, save_dir, sample_rate=1)