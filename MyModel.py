# -*- codeing = utf-8 -*-
# @Author : linxihao
# @File : MyModel.py
# @Software : PyCharm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *


class MyModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MyModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvWithMBA(n_channels, 64)
        self.down1 = EcMBA(64, 128)
        self.down2 = EcMBA(128, 256)
        self.down3 = EcMBA(256, 512)
        # self.down4 = Down(512, 512)
        self.down4 = BnDCN_Context(512,512)
        self.up1 = UpMBA(1024, 256, bilinear)
        self.up2 = UpMBA(512, 128, bilinear)
        self.up3 = UpMBA(256, 64, bilinear)
        self.up4 = UpMBA(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    net = MyModel(n_channels=3, n_classes=17)
    import numpy as np

    # x = torch.from_numpy(np.random.rand(1, 1, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    y = net(x)
    print(y.shape)