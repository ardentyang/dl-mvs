"""
Zhao M, Zhong S, Fu X, et al.
Deep residual shrinkage networks for fault diagnosis[J].
IEEE Transactions on Industrial Informatics, 2019, 16(7): 4681-4690.
"""

import torch
import torch.nn as nn


class SCW(nn.Module):
    """Shrinkage with Channel-Wise thresholds"""
    def __init__(self, num_channel):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(num_channel, num_channel, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(num_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_channel, num_channel, kernel_size=1, stride=1, bias=True)

        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x_abs = torch.abs(x)
        x_avg = self.avgpool(x_abs)

        alpha = self.conv1(x_avg)
        alpha = self.bn1(alpha)
        alpha = self.relu(alpha)
        alpha = self.conv2(alpha)
        alpha = torch.sigmoid(alpha)

        tau = x_avg * alpha

        m = torch.max(x_abs - tau, torch.zeros_like(x))

        return m * torch.sign(x)

        # out = m * torch.sign(x)
        # import matplotlib.pyplot as plt
        # x_tmp = x[0, 0, ].flatten().cpu().detach().numpy()
        # y_tmp = out[0, 0, ].flatten().cpu().detach().numpy()
        # plt.plot(x_tmp, y_tmp, '.')
        # plt.show(block=True)
        #
        # return out


class RSBUCW(nn.Module):
    """Residual Shrinkage Building Unit with Channel-Wise thresholds"""
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        if stride != 1 or in_channel != out_channel:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=True)
            # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.bn1 = nn.BatchNorm1d(out_channel)
            # nn.init.constant_(self.bn1.weight, 1)
            # nn.init.constant_(self.bn1.bias, 0)
            self.downsample = nn.Sequential(self.conv1, self.bn1)
        else:
            self.downsample = None

        self.bn2 = nn.BatchNorm1d(in_channel)
        # nn.init.constant_(self.bn2.weight, 1)
        # nn.init.constant_(self.bn2.bias, 0)
        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        # nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.bn3 = nn.BatchNorm1d(out_channel)
        # nn.init.constant_(self.bn3.weight, 1)
        # nn.init.constant_(self.bn3.bias, 0)
        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        # nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

        self.scw = SCW(num_channel=out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.scw(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class DRSNCW(nn.Module):
    """Deep Residual Shrinkage Network with Channel-Wise thresholds"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        inplanes = 8

        self.conv = nn.Conv1d(in_channel, inplanes, kernel_size=3, stride=2, padding=1, bias=True)

        self.layer1 = RSBUCW(inplanes, inplanes, stride=2)
        self.layer1_1 = RSBUCW(inplanes, inplanes, stride=1)  # *3
        self.layer1_2 = RSBUCW(inplanes, inplanes, stride=1)
        self.layer1_3 = RSBUCW(inplanes, inplanes, stride=1)

        self.layer2 = RSBUCW(inplanes, inplanes * 2, stride=2)
        self.layer2_1 = RSBUCW(inplanes * 2, inplanes * 2, stride=1)  # *3
        self.layer2_2 = RSBUCW(inplanes * 2, inplanes * 2, stride=1)
        self.layer2_3 = RSBUCW(inplanes * 2, inplanes * 2, stride=1)

        self.layer3 = RSBUCW(inplanes * 2, inplanes * 4, stride=2)
        self.layer3_1 = RSBUCW(inplanes * 4, inplanes * 4, stride=1)  # *3
        self.layer3_2 = RSBUCW(inplanes * 4, inplanes * 4, stride=1)
        self.layer3_3 = RSBUCW(inplanes * 4, inplanes * 4, stride=1)

        self.bn = nn.BatchNorm1d(inplanes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(inplanes * 4, out_channel)

        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        # nn.init.constant_(self.bn2.weight, 1)
        # nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)

        x = self.layer2(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        x = self.layer3(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    IN_CHANNEL = 1
    SEQ_LEN = 1024

    OUT_CHANNEL = 10

    def test_scw():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL * 16, SEQ_LEN // 2)
        x_in = x_in.to(DEVICE)

        model = SCW(num_channel=IN_CHANNEL * 16)
        model.to(DEVICE)

        net_out = model(x_in)

    # test_scw()

    def test_rsbu_cw():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL * 16, SEQ_LEN // 2)
        x_in = x_in.to(DEVICE)

        model = RSBUCW(in_channel=IN_CHANNEL * 16, out_channel=IN_CHANNEL * 32, stride=2)
        model.to(DEVICE)

        net_out = model(x_in)
        print(net_out.shape)

    # test_rsbu_cw()


    def test_drsn_cw():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL, SEQ_LEN)
        x_in = x_in.to(DEVICE)

        model = DRSNCW(in_channel=IN_CHANNEL, out_channel=OUT_CHANNEL)
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total-params-of-{model.__class__.__name__}: {total_params}, "
              f"and {total_trainable_params} trainable")

        model.to(DEVICE)

        net_out = model(x_in)
        print(net_out.shape)

    test_drsn_cw()

