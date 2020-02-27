import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channels, reduction_ratio=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_v3(nn.Module):
    def __init__(self):
        super(SE_v3, self).__init__()
        # Initial Block
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2)
        )
        # First SE Block
        self.pre_se_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=48)
        )
        self.se_block_1 = SELayer(channels=48)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        # Second SE Block
        self.pre_se_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96)
        )
        self.se_block_2 = SELayer(channels=96)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        # Decision block
        self.pre_se_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8)
        )
        self.final_se_block = SELayer(channels=8,reduction_ratio=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial Block
        x = self.initial_block(x)
        # First SE Block
        x = self.pre_se_block_1(x)
        x = self.se_block_1(x)
        x = self.maxpool_1(x)
        # Second SE Block
        x = self.pre_se_block_2(x)
        x = self.se_block_2(x)
        x = self.maxpool_2(x)
        # Decision block
        x = self.pre_se_block_3(x)
        x = self.final_se_block(x)
        x = self.avg_pool(x)
        x = self.softmax_1(x)
        return x

class SE_v2(nn.Module):
    def __init__(self):
        super(SE_v3, self).__init__()
        # Initial Block
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2)
        )
        # First SE Block
        self.pre_se_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=48)
        )
        self.se_block_1 = SELayer(channels=48, reduction_ratio=4)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        # Second SE Block
        self.pre_se_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64)
        )
        self.se_block_2 = SELayer(channels=64,reduction_ratio=4)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        # Third SE Block
        self.pre_se_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96)
        )
        self.se_block_3 = SELayer(channels=96,reduction_ratio=4)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)
        # Decision block
        self.pre_se_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8)
        )
        self.final_se_block = SELayer(channels=8,reduction_ratio=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial Block
        x = self.initial_block(x)
        # First SE Block
        x = self.pre_se_block_1(x)
        x = self.se_block_1(x)
        x = self.maxpool_1(x)
        # Second SE Block
        x = self.pre_se_block_2(x)
        x = self.se_block_2(x)
        x = self.maxpool_2(x)
        # Third SE Block
        x = self.pre_se_block_3(x)
        x = self.se_block_3(x)
        x = self.maxpool_3(x)
        # Decision block
        x = self.pre_se_block_4(x)
        x = self.final_se_block(x)
        x = self.avg_pool(x)
        x = self.softmax_1(x)
        return x