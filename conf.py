# -*- coding: utf-8 -*-

vid_feat_path = './data/dynaimic_feat/'
img_feat_path = './data/static_feat/'
train_label_path = './data/final_train.txt'
test_label_path = './data/final_test.txt'
clip_num = 26
image_num = 40

# pretrained_model
i3d_model_path = './pretrained_models/i3d_rgb.pth'

# optimizer
optim_type = 'SGD'
lr = 0.01
momentum = 0.9
weight_decay = 1e-4

# train
epochs = {'Ball': 400, 'Clubs': 300, 'Hoop': 500, 'Ribbon': 300}
batch_size = 32
