import torch
from torch import nn


class DACNN(nn.Module):
    def __init__(self, n_channel=22):
        super().__init__()
        self.n_channel = n_channel
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, kernel_size=1),
            nn.Conv2d(n_channel, n_channel, kernel_size=5),
            nn.Conv2d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm2d(n_channel),
            nn.AdaptiveAvgPool2d((6, 32)),
            nn.ReLU(True),
            nn.Conv2d(n_channel, n_channel // 2, kernel_size=3),
            nn.Conv2d(n_channel // 2, n_channel // 4, kernel_size=3),
            nn.BatchNorm2d(n_channel // 4),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.ReLU(True),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(n_channel // 4 * 16 * 4, 100),
            nn.Dropout(0.5),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, grl_lambda=1.0):
        # 通过扩展(重复)单维来处理单通道输入
        # x = x.expand(x.data.shape[0], 3, image_size, image_size)

        features = self.feature_extractor(x)
        # print(features.shape)
        features = features.view(-1, self.n_channel // 4 * 4 * 16)
        # print(features.shape)

        class_pred = self.class_classifier(features)
        return class_pred


# d = torch.Tensor([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]])
# a = torch.flip(d, [0])
# print(a)
