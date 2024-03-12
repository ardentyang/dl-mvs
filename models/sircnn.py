"""
ref.
Dechen Yao, Hengchang Liu, Jianwei Yang, Xi Li,
A lightweight neural network with strong robustness for bearing fault diagnosis,
Measurement,
Volume 159,
2020,
107756,
"""
import math

import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        expand_planes = in_planes * expansion_factor
        self.conv1 = nn.Conv2d(
            in_planes, expand_planes,
            kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(expand_planes)
        self.relu6 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(
            expand_planes, expand_planes,
            kernel_size=3, stride=stride, padding=1, groups=expand_planes)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(
            expand_planes, out_planes,
            kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu6(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.in_planes == self.out_planes:
            out += x

        return out


class SIRCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        t = [1, 6, 6]
        c = [16, 24, 96]
        s = [1, 2, 1]

        self.conv = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU6(inplace=True)

        self.block1 = InvertedResidualBlock(32, c[0], t[0], s[0])

        self.block2 = InvertedResidualBlock(c[0], c[1], t[1], s[1])
        self.block3 = InvertedResidualBlock(c[1], c[1], t[1], 1)

        self.block4 = InvertedResidualBlock(c[1], c[2], t[2], s[2])
        self.block5 = InvertedResidualBlock(c[2], c[2], t[2], 1)
        self.block6 = InvertedResidualBlock(c[2], c[2], t[2], 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    import torch

    def main():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        seqs_size = 2048
        seqs_chans = 1
        num_classes = 3
        batch_size = 16

        model = SIRCNN(in_channel=seqs_chans, out_channel=num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total-params-of-({type(model)}): {total_params}, "
              f"and {total_trainable_params} trainable")

        model.to(device)

        seqs = torch.randn(batch_size, seqs_chans, seqs_size)
        seqs = seqs.to(device)
        out = model(seqs.view(seqs.size(0), seqs.size(2) // 32, -1).unsqueeze(1))
        print(out.shape)

    main()

