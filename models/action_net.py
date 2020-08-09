# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ContextAttention(nn.Module):
    def __init__(self,  in_size):
        super(ContextAttention, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_size, in_size//2),
            nn.ReLU(),
            nn.Linear(in_size//2, 256),
            nn.ReLU(),
        )
        self.gcn1 = nn.Linear(256, 256)
        self.gcn2 = nn.Linear(256, 256)
        self.att = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, adj):
        x = self.embedding(x)

        cx = torch.relu(self.gcn1(torch.matmul(adj, x)))
        cx = torch.relu(self.gcn2(torch.matmul(adj, cx)))

        x = torch.cat([x, cx], dim=2)
        att = self.att(x)
        x = torch.sum(torch.mul(x, att), dim=1)
        return x, None


class ACTION_NET(nn.Module):
    def __init__(self, clip_num=26, image_num=80):
        super(ACTION_NET, self).__init__()
        self.d_ct = ContextAttention(1024)
        self.s_ct = ContextAttention(2048)

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        t = torch.arange(0, clip_num, dtype=torch.float)
        t1 = t.repeat(clip_num, 1)
        t2 = t1.permute([1, 0])
        dis1 = torch.exp(-1 * torch.abs((t2 - t1) / 1))
        self.adj1 = nn.Parameter(self.normalize(dis1), requires_grad=False)

        t = torch.arange(0, image_num, dtype=torch.float)
        t1 = t.repeat(image_num, 1)
        t2 = t1.permute([1, 0])
        dis2 = torch.exp(-1 * torch.abs((t2 - t1) / 1))
        self.adj2 = nn.Parameter(self.normalize(dis2), requires_grad=False)

    def normalize(self, A, symmetric=True):
        # A = A+I
        # A = A + torch.eye(A.size(0))
        d = A.sum(1)
        if symmetric:
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            # D=D^-1
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), x1.size(1), -1)
        x1, _1 = self.d_ct(x1, self.adj1)

        x2 = x2.view(x2.size(0), x2.size(1), -1)
        x2, _2 = self.s_ct(x2, self.adj2)

        x = torch.cat([x1, x2], dim=1)

        score = self.fc(x).squeeze(dim=1)
        return score
