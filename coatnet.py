import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

from einops import rearrange
from einops.layers.torch import Rearrange

from typing import OrderedDict, Tuple, Optional


def _get_conv_output_shape(input_size: Tuple[int, int], kernel_size: int, stride: int, padding: int):
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


class ConvStem(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, stride, activation=nn.GELU, norm=nn.BatchNorm2d):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm = norm(out_channels)
        self.act = activation()

        self.output_size = _get_conv_output_shape(image_size, 3, stride, 1)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MBConv(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, expand_ratio, se_ratio, stride: int,
                 activation=nn.GELU, norm=nn.BatchNorm2d, dropout=0.0):
        super().__init__()

        proj = nn.ModuleList([])

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            if stride == 2:
                proj.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))
            proj.append(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=1, padding=0, bias=True))
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(in_channels * expand_ratio)
        sqz_channels = max(1, int(mid_channels * se_ratio))

        if dropout:
            self.stochastic_depth = StochasticDepth(dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = norm(in_channels)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            norm_layer=norm,
            activation_layer=activation,
            inplace=None,
        )  # From eqn (5), downsampling is applied here
        _layers["depthwise_conv"] = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_channels,
            norm_layer=norm,
            activation_layer=activation,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(mid_channels, sqz_channels, activation=activation)
        _layers["conv_b"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.layers = nn.Sequential(_layers)
        self.output_size = _get_conv_output_shape(image_size, 1, stride, 0)

    def forward(self, x):
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class RelativeMultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, image_size: Tuple[int, int],
                 head_dim: int = 32, n_heads: Optional[int] = None, dropout: float = 0.0):
        super().__init__()

        assert in_channels % head_dim == 0, f"in_channels: {in_channels} must be divisible by head_dim: {head_dim}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads if n_heads is not None else in_channels // head_dim
        # print(self.n_heads)
        self.head_dim = head_dim
        self.dim = self.n_heads * self.head_dim
        self.h, self.w = image_size

        self.to_qkv = nn.Linear(in_channels, self.dim * 3)
        self.scale_factor = self.dim**-0.5

        self.merge = nn.Linear(self.dim, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.h - 1) * (2 * self.w - 1), self.n_heads), dtype=torch.float32),
        )

        self.register_buffer("relative_position_index", self.get_relative_position_index(self.h, self.w))
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_position_index(self, height, width):
        coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += height - 1
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        return relative_coords.sum(-1).flatten().unsqueeze(1)

    def get_relative_positional_bias(self):
        relative_bias = torch.gather(self.relative_position_bias_table, 0, self.relative_position_index)
        relative_bias = relative_bias.reshape(self.h * self.w, self.h * self.w, -1)
        relative_bias = relative_bias.permute(2, 0, 1)
        relative_bias = relative_bias.unsqueeze(0)
        return relative_bias

    def forward(self, x):
        # B (H W) C
        batch = x.shape[0]

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        score = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale_factor
        pos_bias = self.get_relative_positional_bias()

        score = torch.softmax(score + pos_bias, dim=-1)

        attn = self.dropout(score)
        attn = torch.matmul(attn, v)

        out = attn.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch, -1, self.dim)

        out = self.merge(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, activation=nn.GELU, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConvTransformer(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, head_dim: int = 32, n_heads: Optional[int] = None,
                 expand_ratio: int = 4, activation=nn.GELU, norm=nn.LayerNorm, dropout=0.0, downsampling=False):
        super().__init__()
        hidden_dim = int(out_channels * expand_ratio)

        proj = nn.ModuleList([])
        attn = nn.ModuleList([])
        ffn = nn.ModuleList([])

        attn.append(norm((in_channels,)+image_size))  # pre-norm

        if downsampling or in_channels != out_channels:
            if downsampling:
                proj.append(nn.MaxPool2d(3, 2, 1))
                attn.append(nn.MaxPool2d(3, 2, 1))
                image_size = _get_conv_output_shape(image_size, 3, 2, 1)
            proj.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True))
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        h, w = image_size

        attn.append(Rearrange('b c h w -> b (h w) c'))
        attn.append(RelativeMultiHeadSelfAttention(in_channels, out_channels, image_size, head_dim, n_heads, dropout))
        attn.append(Rearrange('b (h w) c -> b c h w', h=h, w=w))

        ffn.append(norm((out_channels,)+image_size))  # pre-norm
        ffn.append(Rearrange('b c h w -> b (h w) c'))
        ffn.append(FFN(out_channels, hidden_dim, activation, dropout))
        ffn.append(Rearrange('b (h w) c -> b c h w', h=h, w=w))

        self.attn = nn.Sequential(*attn)
        self.ffn = nn.Sequential(*ffn)

        self.output_size = image_size

    def forward(self, x):
        x = self.proj(x) + self.attn(x)
        x = x + self.ffn(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, stage_configs, num_classes=1000,
                 activation=nn.GELU, expand_ratio=4, se_ratio=0.25, dropout_rate=0.0,
                 head_dim=32, n_heads=None):
        super().__init__()
        self.activation = activation
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.dropout_rate = dropout_rate

        _layers = OrderedDict()
        for i, (b_type, n_block, out_channels, downsample) in enumerate(stage_configs):
            for j in range(n_block):
                downsample = downsample and j == 0
                stride = 2 if downsample else 1
                if b_type.upper() == 'C':
                    layer = ConvStem(
                        image_size, in_channels, out_channels, stride, activation=self.activation)
                elif b_type.upper() == 'M':
                    layer = MBConv(
                        image_size, in_channels, out_channels, self.expand_ratio, self.se_ratio, stride,
                        activation=self.activation, dropout=self.dropout_rate)
                elif b_type.upper() == 'T':
                    layer = ConvTransformer(
                        image_size, in_channels, out_channels, head_dim=head_dim, n_heads=n_heads,
                        expand_ratio=self.expand_ratio, activation=self.activation, dropout=dropout_rate,
                        downsampling=downsample)
                else:
                    layer = nn.Identity()
                _layers['s{}_l{}'.format(i,j)] = layer
                in_channels = out_channels
                image_size = layer.output_size

        self.stages = nn.Sequential(_layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.stages(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.dropout(x)
        x = self.fc(x)
        return x


def coatnet_0():
    stage_configs = [
        ('C', 2, 64,  True),
        ('M', 2, 96,  True),
        ('M', 3, 192, True),
        ('T', 5, 384, True),
        ('T', 2, 768, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, n_heads=12)

def coatnet_1():
    stage_configs = [
        ('C', 2,  64,  True),
        ('M', 2,  96,  True),
        ('M', 6,  192, True),
        ('T', 14, 384, True),
        ('T', 2,  768, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, n_heads=12)


def coatnet_2():
    stage_configs = [
        ('C', 2,  128,  True),
        ('M', 2,  128,  True),
        ('M', 6,  256,  True),
        ('T', 14, 512,  True),
        ('T', 2,  1024, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, n_heads=12)


def coatnet_3():
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 6,  384,  True),
        ('T', 14, 768,  True),
        ('T', 2,  1536, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, n_heads=20)


def coatnet_4():
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 12, 384,  True),
        ('T', 28, 768,  True),
        ('T', 2,  1536, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, n_heads=20)


def coatnet_5():
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  256,  True),
        ('M', 12, 512,  True),
        ('T', 28, 1280, True),
        ('T', 2,  2048, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, head_dim=64, n_heads=16)


def coatnet_6():
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 4,  384,  True),
        ('M', 8,  768,  True),
        ('T', 42, 1536, False),
        ('T', 2,  2048, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, head_dim=128, n_heads=16)


def coatnet_7():
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  256,  True),
        ('M', 4,  512,  True),
        ('M', 8,  1024, True),
        ('T', 42, 2048, False),
        ('T', 2,  3072, True),
    ]
    return CoAtNet((224, 224), 3, stage_configs, num_classes=1000, head_dim=128, n_heads=16)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    # net = coatnet_1()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_2()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_3()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_4()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_5()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_6()
    # out = net(img)
    # print(out.shape, count_parameters(net))

    # net = coatnet_7()
    # out = net(img)
    # print(out.shape, count_parameters(net))
