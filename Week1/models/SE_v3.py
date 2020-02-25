import torch
from torch import nn


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return t


class SE_v3(nn.Module):

    def __init__(self, input_shape=(256,256,3), num_classes=10):
        super(SE_v3, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        # First block
        self.conv_1_1 = nn.Conv2d(in_channels=self.input_shape, out_channels=32, kernel_size=5, strides=2, padding=0)
        self.relu_1_1 = nn.ReLU()
        self.batchnorm_1_1 = nn.BatchNorm2d(num_features=32)
        self.maxpool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second block
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, strides=1, padding=0)
        self.relu_2_1 = nn.ReLU()
        self.batchnorm_2_1 = nn.BatchNorm2d(num_features=64)
        self.avgpool_2_1 = nn.AvgPool2d(kernel_size=1)
        self.dense_2_1 = nn.Linear(in_features=64, out_features=16)
        self.relu_2_2 = nn.ReLU()
        self.dense_2_2 = nn.Linear(in_features=16, out_features=64)
        self.sigmoid_2_1 = nn.Sigmoid()
        self.multiply_2_1 = Multiply()
        self.maxpool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Third block
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, strides=1, padding=0)
        self.relu_3_1 = nn.ReLU()
        self.batchnorm_3_1 = nn.BatchNorm2d(num_features=128)
        self.avgpool_3_1 = nn.AvgPool2d(kernel_size=1)
        self.dense_3_1 = nn.Linear(in_features=128, out_features=32)
        self.relu_3_2 = nn.ReLU()
        self.dense_3_2 = nn.Linear(in_features=32, out_features=128)
        self.sigmoid_3_1 = nn.Sigmoid()
        self.multiply_3_1 = Multiply()
        self.maxpool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fourth block
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, strides=1, padding=0)
        self.relu_4_1 = nn.ReLU()
        self.batchnorm_4_1 = nn.BatchNorm2d(num_features=128)
        self.avgpool_4_1 = nn.AvgPool2d(kernel_size=1)
        self.dense_4_1 = nn.Linear(in_features=128, out_features=32)
        self.relu_4_2 = nn.ReLU()
        self.dense_4_2 = nn.Linear(in_features=32, out_features=128)
        self.sigmoid_4_1 = nn.Sigmoid()
        self.multiply_4_1 = Multiply()
        self.maxpool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Decision block
        self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, strides=1, padding=1)
        self.relu_5_1 = nn.ReLU()
        self.batchnorm_5_1 = nn.BatchNorm2d(num_features=8)
        self.avgpool_5_1 = nn.AvgPool2d(kernel_size=1)
        self.dense_5_1 = nn.Linear(in_features=8, out_features=8)
        self.relu_5_2 = nn.ReLU()
        self.dense_5_2 = nn.Linear(in_features=8, out_features=8)
        self.sigmoid_5_1 = nn.Sigmoid()
        self.multiply_5_1 = Multiply()
        self.avgpool_5_2 = nn.AvgPool2d(kernel_size=1)
        self.softmax_5_1 = nn.Softmax(dim=1)

    def forward(self, x):
        # First block
        x = self.conv_1_1(x)
        x = self.relu_1_1(x)
        x = self.batchnorm_1_1(x)
        x = self.maxpool_1_1(x)
        # Second block
        x = self.conv_2_1(x)
        x = self.relu_2_1(x)
        x = self.batchnorm_2_1(x)
        y = self.avgpool_2_1(x).view(1, 1, 64)
        y = self.dense_2_1(y)
        y = self.relu_2_2(y)
        y = self.dense_2_2(y)
        y = self.sigmoid_2_1(y)
        x = self.multiply_2_1([x, y])
        x = self.maxpool_2_1(x)
        # Third block
        x = self.conv_3_1(x)
        x = self.relu_3_1(x)
        x = self.batchnorm_3_1(x)
        y = self.avgpool_3_1(x).view(1, 1, 128)
        y = self.dense_3_1(y)
        y = self.relu_3_2(y)
        y = self.dense_3_2(y)
        y = self.sigmoid_3_1(y)
        x = self.multiply_3_1([x, y])
        x = self.maxpool_3_1(x)
        # Fourth block
        x = self.conv_4_1(x)
        x = self.relu_4_1(x)
        x = self.batchnorm_4_1(x)
        y = self.avgpool_4_1(x).view(1, 1, 128)
        y = self.dense_4_1(y)
        y = self.relu_4_2(y)
        y = self.dense_4_2(y)
        y = self.sigmoid_4_1(y)
        x = self.multiply_4_1([x, y])
        x = self.maxpool_4_1(x)
        # Decision block
        x = self.conv_5_1(x)
        x = self.relu_5_1(x)
        x = self.batchnorm_5_1(x)
        y = self.avgpool_5_1(x).view(1, 1, 8)
        y = self.dense_5_1(y)
        y = self.relu_5_2(y)
        y = self.dense_5_2(y)
        y = self.sigmoid_5_1(y)
        x = self.multiply_5_1([x, y])
        x = self.avgpool_5_2(x)
        x = self.softmax_5_1(x)

        return x
