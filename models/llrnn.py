"""
ref.
Liu, W.; Guo, P.; Ye, L. A Low-Delay Lightweight Recurrent Neural Network (LLRNN)
for Rotating Machinery Fault Diagnosis. Sensors 2019, 19, 3109.
"""

import torch
import torch.nn as nn


class LLRNN(nn.Module):
    """Low-delay lightweight recurrent neural network"""
    def __init__(
            self, in_dims, cell_dims, out_dims,
            num_timesteps, output_activation=None):
        super().__init__()

        self.cell_dims = cell_dims
        self.num_timesteps = num_timesteps

        kernel_data = torch.zeros(in_dims, 2 * cell_dims, dtype=torch.get_default_dtype())
        recurrent_kernel = torch.zeros(cell_dims, 2 * cell_dims, dtype=torch.get_default_dtype())
        recurrent_bias = torch.zeros(2 * cell_dims, dtype=torch.get_default_dtype())
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.output_dense = nn.Linear(cell_dims, out_dims)
        self.output_activation = output_activation

        # initialization
        self.kernel = nn.Parameter(kernel_data)
        torch.nn.init.xavier_uniform_(self.kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        torch.nn.init.xavier_uniform_(self.recurrent_kernel)
        recurrent_bias[:cell_dims] = torch.log(
            1. + ((num_timesteps - 1) - 1.) * torch.rand(cell_dims)).to(dtype=torch.float32)
        self.recurrent_bias = nn.Parameter(recurrent_bias)

    def forward(self, inputs):
        h_state = torch.zeros(inputs.size(0), self.cell_dims, dtype=torch.get_default_dtype()).to(inputs.device)
        c_state = torch.zeros(inputs.size(0), self.cell_dims, dtype=torch.get_default_dtype()).to(inputs.device)

        num_timesteps = inputs.size(1)
        assert self.num_timesteps == num_timesteps

        for t in range(num_timesteps):
            ip = inputs[:, t, :]

            z = torch.mm(ip, self.kernel)
            z += torch.mm(h_state, self.recurrent_kernel) + self.recurrent_bias

            z0 = z[:, : self.cell_dims]
            z1 = z[:, self.cell_dims: self.cell_dims * 2]

            f = self.sigmoid(z0)
            c = f * c_state + (1. - f) * self.tanh(z1)

            h = c

            h_state = h
            c_state = c

        preds = self.output_dense(h_state)

        if self.output_activation is not None:
            preds = self.output_activation(preds)

        return preds


if __name__ == '__main__':

    def main():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        seqs_size = 2048
        seqs_chans = 1
        num_classes = 3
        d_x = 64
        d_h = 128
        batch_size = 16

        model = LLRNN(
            in_dims=d_x, cell_dims=d_h, out_dims=num_classes,
            num_timesteps=seqs_size // d_x,
            output_activation=None)
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total-params-of-({type(model)}): {total_params}, "
              f"and {total_trainable_params} trainable")

        model.to(device)

        seqs = torch.randn(batch_size, seqs_chans, seqs_size)
        seqs = seqs.to(device)
        out = model(seqs.view(seqs.size(0), seqs_size // d_x, d_x))
        print(out.shape)

    main()

