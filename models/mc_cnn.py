"""
ref.
Huang W, Cheng J, Yang Y, et al.
An improved deep convolutional neural network with multi-scale information for bearing fault diagnosis[J].
Neurocomputing, 2019, 359: 77-92.
"""

import torch
import torch.nn as nn


class MCCNN(nn.Module):
    """Multi-scale Cascade Convolutional Neural Network (MC-CNN)"""
    def __init__(self, in_channel, out_channel):
        super(MCCNN, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channel, 1, kernel_size=100, stride=1, bias=True)
        self.conv1_2 = nn.Conv1d(in_channel, 1, kernel_size=200, stride=1, bias=True)
        self.conv1_3 = nn.Conv1d(in_channel, 1, kernel_size=300, stride=1, bias=True)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()

        self.conv2 = nn.Conv1d(1, 8, kernel_size=8, stride=2, bias=True)
        self.bn2 = nn.BatchNorm1d(8)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(8, 8, kernel_size=32, stride=4, bias=True)
        self.bn3 = nn.BatchNorm1d(8)

        self.conv4 = nn.Conv1d(8, 8, kernel_size=16, stride=2, bias=True)
        self.bn4 = nn.BatchNorm1d(8)

        self.fc1 = nn.Linear(8 * 14, 112)
        self.bn5 = nn.BatchNorm1d(112)

        self.fc2 = nn.Linear(112, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.cat([self.conv1_1(x), self.conv1_2(x), self.conv1_3(x)], dim=-1)
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.sig(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sig(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sig(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sig(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.sig(x)

        x = self.fc2(x)

        return x


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    IN_CHANNEL = 1
    SEQ_LEN = 1024

    STEPS = 64

    OUT_CHANNEL = 10


    def test_msc_cnn():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL, SEQ_LEN)
        x_in = x_in.to(DEVICE)

        model = MscCNN(in_channel=IN_CHANNEL, out_channel=OUT_CHANNEL)
        model.to(DEVICE)

        net_out = model(x_in)
        print(net_out.shape)


    test_msc_cnn()

