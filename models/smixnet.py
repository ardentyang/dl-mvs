"""
ref.
Jie Wu, Tang Tang, Ming Chen, Yi Wang, Kesheng Wang,
A study on adaptation lightweight architecture based deep learning models for bearing fault diagnosis under varying working conditions,
Expert Systems with Applications,
Volume 160,
2020,
113710
"""

import torch
import torch.nn as nn


# class SMobileNetV2(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x
#
#
# class SMobileNetV3(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.Module):
    """ Mixed Depthwise Convolution
    Paper: MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)
    Based on
     - MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
     - PyTorch Mixed Convolution:
      https://github.com/huggingface/pytorch-image-models/blob/36449617ffea4487ce27819554f8bc13b5df335d/timm/layers/mixed_conv2d.py
    """
    def __init__(self, num_channels, kernel_size=None,
                 stride=1, dilation=1, **kwargs):
        super().__init__()
        if kernel_size is None:
            kernel_size = [3, 5]
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        ch_splits = _split_channels(num_channels, num_groups)
        self.mixedconv = nn.ModuleList()
        for idx, (k, ch) in enumerate(zip(kernel_size, ch_splits)):
            self.mixedconv.append(
                nn.Conv2d(
                    ch, ch, k, stride=stride,
                    padding=k // 2, dilation=dilation, groups=ch, **kwargs))
        self.splits = ch_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.mixedconv)]
        x = torch.cat(x_out, 1)
        return x


class MixConvBlock(nn.Module):
    """
    Based on InvertedResidual from
      https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/mobilenet.py
    """
    def __init__(self, inplanes, planes, mixed_kernels=None, stride=1, expand_ratio=1):
        super().__init__()
        if mixed_kernels is None:
            mixed_kernels = 3

        hidden_dim = int(round(inplanes * expand_ratio))
        # pw conv
        self.conv1 = nn.Conv2d(inplanes, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.swish = nn.SiLU(inplace=True)
        # mixed dw conv
        self.conv2 = MixedConv2d(hidden_dim, kernel_size=mixed_kernels, stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        # pw conv
        self.conv3 = nn.Conv2d(hidden_dim, planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes)

        self.use_res_connect = stride == 1 and inplanes == planes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.swish(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.swish(out)

        if self.use_res_connect:
            return x + out
        else:
            return out


class SMixNet(nn.Module):
    """Simplified MixNet-small
    Based on
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_builder.py
    """
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        last_planes = 96

        # stem
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.mcb1 = MixConvBlock(16, 32, mixed_kernels=3, stride=2, expand_ratio=6)
        self.mcb2 = MixConvBlock(32, 64, mixed_kernels=3, stride=2, expand_ratio=6)
        self.mcb3 = MixConvBlock(64, 96, mixed_kernels=[3, 5], stride=2, expand_ratio=6)
        self.mcb4 = MixConvBlock(96, last_planes, mixed_kernels=[3, 5, 7], stride=1, expand_ratio=6)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_planes, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mcb1(x)
        x = self.mcb2(x)
        x = self.mcb3(x)
        x = self.mcb4(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':

    def code_test_mixed_conv():
        mixed_conv = MixedConv2d(64, kernel_size=3, stride=2)
        a = torch.randn(8, 64, 16, 16)
        out = mixed_conv(a)
        print(out.shape)

    # code_test_mixed_conv()


    def code_test_mix_conv_block():
        mix_conv_block = MixConvBlock(64, 96, mixed_kernels=[3, 5], stride=2, expand_ratio=6)
        a = torch.randn(16, 64, 8, 8)
        out = mix_conv_block(a)
        print(out.shape)

    # code_test_mix_conv_block()

    def main():
        model = SMixNet(in_channels=1, out_channels=10)
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total-params-of-({type(model)}): {total_params}, "
              f"and {total_trainable_params} trainable")
        a = torch.randn(16, 1, 64, 64)
        out = model(a)
        print(out.shape)

    main()

