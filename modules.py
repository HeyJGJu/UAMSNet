# -*- codeing = utf-8 -*-
# @Author : linxihao
# @File : modules.py
# @Software : PyCharm
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)

class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out)
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()