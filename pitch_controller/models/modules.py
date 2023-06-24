# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from .base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x):
        output = self.block(x)
        return output


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        output = h + self.res_conv(x)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32, q_norm=True):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)
        self.q_norm = q_norm

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        if self.q_norm:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(BaseModule):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class PitchPosEmb(BaseModule):
    def __init__(self, dim, flip_sin_to_cos=False, downscale_freq_shift=0):
        super(PitchPosEmb, self).__init__()
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, x):
        # B * L
        b, l = x.shape
        x = rearrange(x, 'b l -> (b l)')
        emb = get_timestep_embedding(
            x,
            self.dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        emb = rearrange(emb, '(b l) d -> b d l', b=b, l=l)
        return emb


class TimbreBlock(BaseModule):
    def __init__(self, out_dim):
        super(TimbreBlock, self).__init__()
        base_dim = out_dim // 4

        self.block11 = torch.nn.Sequential(torch.nn.Conv2d(1, 2 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.block12 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 2 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.block21 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 4 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.block22 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 4 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.block31 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 8 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.block32 = torch.nn.Sequential(torch.nn.Conv2d(4 * base_dim, 8 * base_dim,
                                                           3, 1, 1),
                                           torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                                           torch.nn.GLU(dim=1))
        self.final_conv = torch.nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(self, x):
        y = self.block11(x)
        y = self.block12(y)
        y = self.block21(y)
        y = self.block22(y)
        y = self.block31(y)
        y = self.block32(y)
        y = self.final_conv(y)

        return y.sum((2, 3)) / (y.shape[2] * y.shape[3])