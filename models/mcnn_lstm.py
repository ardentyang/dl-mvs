"""
Chen X, Zhang B, Gao D.
Bearing fault diagnosis base on multi-scale CNN and LSTM model[J].
Journal of Intelligent Manufacturing, 2021, 32(4): 971-987.

implemented by Xiaoqi Yang, 2022
"""


import torch
import torch.nn as nn


class MCNNLSTM(nn.Module):
    """Multi-scale Convolutional Neural Network and Long Short-Term Memory"""
    def __init__(self, in_channel, out_channel):
        super(MCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 50, kernel_size=20, stride=2, bias=True)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv1d(50, 30, kernel_size=10, stride=2, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channel, 50, kernel_size=6, stride=1, bias=True)
        self.conv4 = nn.Conv1d(50, 40, kernel_size=6, stride=1, bias=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(40, 30, kernel_size=6, stride=1, bias=True)
        self.conv6 = nn.Conv1d(30, 30, kernel_size=6, stride=2, bias=True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(input_size=30, hidden_size=60)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=30)

        self.fc = nn.Linear(30, out_channel)

    def forward(self, x):
        x = x[:, :, ::4]  # 1024 // 4 = 256

        x1 = self.conv1(x)  # 119
        x1 = self.tanh(x1)
        x1 = self.conv2(x1)  # 57
        x1 = self.tanh(x1)
        x1 = self.maxpool1(x1)  # 28
        # print(x1.shape)

        x2 = self.conv3(x)  # 251
        x2 = self.tanh(x2)
        x2 = self.conv4(x2)  # 246
        x2 = self.tanh(x2)
        x2 = self.maxpool2(x2)  # 123
        x2 = self.conv5(x2)  # 118
        x2 = self.tanh(x2)
        x2 = self.conv6(x2)  # 57
        x2 = self.tanh(x2)
        x2 = self.maxpool3(x2)  # 28
        # print(x2.shape)

        x = x1 * x2  # (batch_size, 30, 28)

        x = x.permute(2, 0, 1)  # (28, batch_size, 30)
        x, _ = self.lstm1(x, )  # (28, batch_size, 60)
        x, _ = self.lstm2(x, )  # (28, batch_size, 30)
        x = x[-1]

        x = self.fc(x)

        return x


def compute_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, total_trainable_params


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    IN_CHANNEL = 1
    SEQ_LEN = 1024

    STEPS = 64

    OUT_CHANNEL = 10

    def test_mcnn_lstm():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL, SEQ_LEN)
        x_in = x_in.to(DEVICE)

        model = MCNNLSTM(in_channel=IN_CHANNEL, out_channel=OUT_CHANNEL)
        model.to(DEVICE)

        net_out = model(x_in)
        print(net_out.shape)

        _, n_params = compute_num_params(model)
        print(n_params)

    test_mcnn_lstm()

