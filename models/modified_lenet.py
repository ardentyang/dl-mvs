"""
ref.
Wen L, Li X, Gao L, et al.
A new convolutional neural network-based data-driven fault diagnosis method[J].
IEEE Transactions on Industrial Electronics, 2017, 65(7): 5990-5998.
"""

import torch
import torch.nn as nn


class ModifiedLeNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 4 * 256 // 4, 2560 // 4)
        self.fc2 = nn.Linear(2560 // 4, 768 // 4)
        self.fc3 = nn.Linear(768 // 4, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = self.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = self.relu(self.conv3(out))
        out = self.maxpool3(out)
        out = self.relu(self.conv4(out))
        out = self.maxpool4(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc3(out)


if __name__ == '__main__':
    model = ModifiedLeNet(in_channel=1, out_channel=10)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total-params-of-({type(model)}): {total_params}, "
          f"and {total_trainable_params} trainable")

