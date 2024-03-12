"""
ref.
Zhang W, Li C, Peng G, et al. A deep convolutional neural network with new training methods
for bearing fault diagnosis under noisy environment and different working load[J].
Mechanical Systems and Signal Processing, 2018, 100: 439-453.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiChildBlock(nn.Module):
    def __init__(self, inplanes, planes, k, s, p):
        super(TiChildBlock, self).__init__()
        self.conv = nn.Conv1d(inplanes, planes,
                              kernel_size=k, stride=s, padding=p,
                              bias=True)
        self.bn = nn.BatchNorm1d(planes)

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bn.weight, 0.5)
        # nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = F.max_pool1d(out, 2)
        return out


class TICNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TICNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channel, 16, kernel_size=64, stride=4, padding=30,  # (1, 16, 64, 8, 28)
            bias=True
        )
        self.dropout = nn.Dropout(p=0.4)
        self.bn = nn.BatchNorm1d(num_features=16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bn.weight, 0.5)
        # nn.init.constant_(self.bn.bias, 0)

        self.layer = nn.Sequential(
            TiChildBlock(16, 32, 3, 1, 1),
            TiChildBlock(32, 64, 3, 1, 1),
            TiChildBlock(64, 64, 3, 1, 1),
            TiChildBlock(64, 64, 3, 1, 1),
            TiChildBlock(64, 64, 3, 1, 0),
        )

        self.fc1 = nn.Linear(64 * 3, 100)
        self.fc2 = nn.Linear(100, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer(out)

        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))

        return self.fc2(out)


if __name__ == '__main__':
    model = TICNN(in_channel=1, out_channel=50)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total-params-of-({type(model)}): {total_params}, "
          f"and {total_trainable_params} trainable")

