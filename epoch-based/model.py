# -*-coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.covs = nn.Sequential(
            nn.BatchNorm2d(22, affine=True),
            nn.Conv2d(22, 22, (7, 3), padding=1, groups=22),
            nn.BatchNorm2d(22, affine=True),
            nn.ReLU(),
            nn.AvgPool2d((5, 2)),
            nn.Conv2d(22, 22, (3, 3), groups=22),
            nn.BatchNorm2d(22, affine=True),
            nn.ReLU(),
            nn.AvgPool2d((3, 2)),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(22, 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y1 = self.covs(x1)
        y1_gap = self.gap(y1)
        y1 = self.fc(y1_gap)
        y2 = self.covs(x2)
        y2_gap = self.gap(y2)
        y2 = self.fc(y2_gap)
        return y1, y2

#
# d = torch.randn((128, 22, 64, 32))
# m = SiameseNetwork()
# y1, y2 = m(d, d)
# print(y1.shape)
# f, p = profile(m, (d, d))
# print(f)  # 7737374720.0
# print(p)  # 19351.0
