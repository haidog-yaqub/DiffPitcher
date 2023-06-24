import torch
import torch.nn as nn

import math
from einops import rearrange


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, q_norm=True):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv1d(hidden_dim, dim, 1)
        self.q_norm = q_norm

    def forward(self, x):
        # b, l, c = x.shape
        x = x.permute(0, 2, 1)
        # b, c, l = x.shape

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l',
                            heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        if self.q_norm:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l',
                        heads=self.heads)
        return self.to_out(out).permute(0, 2, 1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, layer_norm_first=True):
        super(TransformerBlock, self).__init__()
        dim_head = dim//n_heads
        self.attention = LinearAttention(dim, heads=n_heads, dim_head=dim_head)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.feed_forward = nn.Sequential(nn.Linear(dim, dim*2),
                                          nn.SiLU(),
                                          nn.Linear(dim*2, dim))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.layer_norm_first = layer_norm_first

    def forward(self, x):
        nx = self.norm1(x)
        x = x + self.dropout1(self.attention(nx))
        nx = self.norm2(x)
        nx = x + self.dropout2(self.feed_forward(nx))
        # attention_out = self.attention(x)
        # attention_residual_out = attention_out + x
        # # print(attention_residual_out.shape)
        # norm1_out = self.dropout1(self.norm1(attention_residual_out))
        #
        # feed_fwd_out = self.feed_forward(norm1_out)
        # feed_fwd_residual_out = feed_fwd_out + norm1_out
        # norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return nx


class PitchFormer(nn.Module):
    def __init__(self, n_mels, hidden_size, attn_layers=4):
        super(PitchFormer, self).__init__()

        self.sp_linear = nn.Sequential(nn.Conv1d(n_mels, hidden_size, kernel_size=1),
                                       nn.SiLU(),
                                       nn.Conv1d(hidden_size, hidden_size//2, kernel_size=1)
                                       )

        self.midi_linear = nn.Sequential(nn.Conv1d(1, hidden_size, kernel_size=1),
                                         nn.SiLU(),
                                         nn.Conv1d(hidden_size, hidden_size//2, kernel_size=1),
                                         )

        self.hidden_size = hidden_size

        self.pos_conv = nn.Conv1d(hidden_size, hidden_size,
                                  kernel_size=63,
                                  padding=31,
                                  )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (63 * hidden_size))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, nn.SiLU())

        self.attn_block = nn.ModuleList([TransformerBlock(hidden_size, 4) for i in range(attn_layers)])

        # self.silu = nn.SiLU()

        self.linear = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.SiLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, midi, sp):
        midi = midi.unsqueeze(1)
        midi = self.midi_linear(midi)
        sp = self.sp_linear(sp)

        x = torch.cat([midi, sp], dim=1)

        # position encoding
        x_conv = self.pos_conv(x)
        x = x + x_conv

        # x = self.silu(x)
        x = x.permute(0, 2, 1)
        for layer in self.attn_block:
            x = layer(x)

        x = self.linear(x)

        return x.squeeze(-1)


if __name__ == '__main__':

    model = PitchFormer(100, 256)

    x = torch.rand((4, 64))
    sp = torch.rand((4, 100, 64))
    midi = torch.rand((4, 64))

    y = model(midi, sp)