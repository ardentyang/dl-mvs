"""
ref.
Li X, Zhang W, Ding Q.
Understanding and improving deep learning-based rolling bearing fault diagnosis with attention mechanism[J].
Signal processing, 2019, 161: 136-154.
"""

import torch
import torch.nn as nn


class BiLSTMAttn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BiLSTMAttn, self).__init__()
        N_input = 1024
        self.N_seg = 4
        N_sub = N_input // self.N_seg

        F_N = 3
        F_L = 5

        self.conv1 = nn.Conv1d(in_channel, F_N, kernel_size=F_L, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(F_N, F_N, kernel_size=F_L, stride=1, padding=2, bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bilstm = nn.LSTM(input_size=F_N * N_sub // 2, hidden_size=16, bidirectional=True)
        self.fc1 = nn.Linear(32, 32)

        self.attn_conv = nn.Conv1d(32, 1, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(16 * 2, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, F_N, N_input // 2)

        x = x.reshape(x.size(0), -1, self.N_seg)  # (batch_size, F_N * N_sub // 2, N_seg)
        x = x.permute(2, 0, 1)
        x, _ = self.bilstm(x, )  # (seq_len, batch_size, num_directions * hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, N_seg, 2 * 16)
        x = self.fc1(x)
        r = self.relu(x)

        r = r.permute(0, 2, 1)  # (batch_size, 2 * 16, N_seg)
        u = self.leaky_relu(self.attn_conv(r))  # (batch_size, 1, N_seg)
        m, _ = torch.max(u, dim=-1, keepdim=True)
        u_m = u - m
        alpha = torch.softmax(u_m, dim=-1)
        alpha = alpha.permute(0, 2, 1)  # (batch_size, N_seg, 1)
        v = torch.matmul(r, alpha)

        v = v.view(v.size(0), -1)
        return self.fc2(v)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    IN_CHANNEL = 1
    SEQ_LEN = 1024

    OUT_CHANNEL = 10


    def test_bilstm_attn():
        x_in = torch.randn(BATCH_SIZE, IN_CHANNEL, SEQ_LEN)
        x_in = x_in.to(DEVICE)

        model = BiLSTMAttn(in_channel=IN_CHANNEL, out_channel=OUT_CHANNEL)
        model.to(DEVICE)

        net_out = model(x_in)
        print(net_out.shape)

    test_bilstm_attn()

