import math
import torch

from .base import BaseModule
from .modules import Mish, Upsample, Downsample, Rezero, Block, ResnetBlock
from .modules import LinearAttention, Residual, Timesteps, TimbreBlock, PitchPosEmb

from einops import rearrange


class UNetVC(BaseModule):
    def __init__(self,
                 dim_base,
                 dim_cond,
                 use_ref_t,
                 use_embed,
                 dim_embed=256,
                 dim_mults=(1, 2, 4),
                 pitch_type='bins'):

        super(UNetVC, self).__init__()
        self.use_ref_t = use_ref_t
        self.use_embed = use_embed
        self.pitch_type = pitch_type

        dim_in = 2

        # time embedding
        self.time_pos_emb = Timesteps(num_channels=dim_base,
                                      flip_sin_to_cos=True,
                                      downscale_freq_shift=0)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4),
                                       Mish(), torch.nn.Linear(dim_base * 4, dim_base))

        # speaker embedding
        timbre_total = 0
        if use_ref_t:
            self.ref_block = TimbreBlock(out_dim=dim_cond)
            timbre_total += dim_cond
        if use_embed:
            timbre_total += dim_embed

        if timbre_total != 0:
            self.timbre_block = torch.nn.Sequential(
                torch.nn.Linear(timbre_total, 4 * dim_cond),
                Mish(),
                torch.nn.Linear(4 * dim_cond, dim_cond))

        if use_embed or use_ref_t:
            dim_in += dim_cond

        self.pitch_pos_emb = PitchPosEmb(dim_cond)
        self.pitch_mlp = torch.nn.Sequential(
            torch.nn.Conv1d(dim_cond, dim_cond * 4, 1, stride=1),
            Mish(),
            torch.nn.Conv1d(dim_cond * 4, dim_cond, 1, stride=1), )
        dim_in += dim_cond

        # pitch embedding
        if self.pitch_type == 'bins':
            print('using mel bins for f0')
        elif self.pitch_type == 'log':
            print('using log bins f0')

        dims = [dim_in, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # blocks
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(self, x, mean, f0, t, ref=None, embed=None):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        if len(t.shape) == 0:
            t = t * torch.ones(x.shape[0], dtype=t.dtype, device=x.device)

        t = self.time_pos_emb(t)
        t = self.mlp(t)

        x = torch.stack([x, mean], 1)

        # if self.pitch_type == 'bins':
        f0 = f0
        f0 = self.pitch_pos_emb(f0)
        f0 = self.pitch_mlp(f0)
        f0 = f0.unsqueeze(2)
        f0 = torch.cat(x.shape[2] * [f0], 2)

        timbre = None
        if self.use_ref_t:
            ref = torch.stack([ref], 1)
            timbre = self.ref_block(ref)
        if self.use_embed:
            if timbre is not None:
                timbre = torch.cat([timbre, embed], 1)
            else:
                timbre = embed
        if timbre is None:
            # raise Exception("at least use one timbre condition")
            condition = f0
        else:
            timbre = self.timbre_block(timbre).unsqueeze(-1).unsqueeze(-1)
            timbre = torch.cat(x.shape[2] * [timbre], 2)
            timbre = torch.cat(x.shape[3] * [timbre], 3)
            condition = torch.cat([f0, timbre], 1)

        x = torch.cat([x, condition], 1)

        hiddens = []
        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_block(x)
        output = self.final_conv(x)

        return output.squeeze(1)