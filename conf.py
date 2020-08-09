# -*- coding: utf-8 -*-

dynamic_feat_path = './data/dynamic_feat/'
static_feat_path = './data/static_feat/'
train_label_path = './data/train.txt'
test_label_path = './data/test.txt'
clip_num = 26
image_num = 80

# optimizer
lr = 0.01
momentum = 0.9
weight_decay = 1e-4

# train
epochs = {'Ball': 400, 'Clubs': 300, 'Hoop': 500, 'Ribbon': 300}
batch_size = 32
