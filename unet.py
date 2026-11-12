"""A compact time-conditioned U-Net (DDPM style) for 64x64 images.

What matters for flow matching (and what the earlier model got wrong):
  * the network must know *where on the path* it is. t in [0, 1] is multiplied by 1000
    before the sinusoidal embedding, so that its many frequencies actually vary over the
    unit interval, and an MLP maps it to a 4*ch vector that every residual block uses to
    scale and shift its normalised features ("FiLM" / adaptive group norm).
  * GroupNorm instead of BatchNorm: the output for one image must not depend on the other
    images in the batch, nor on running statistics gathered at a mix of times t.
  * skip connections at every resolution, including the full 64x64 one, plus
    self-attention at 16x16 and 8x8 so that far-apart parts of a face can agree.
  * the last convolution is zero-initialised, so training starts from v = 0.

Optional conditioning image (`cond_channels > 0`): it is concatenated to the input as
extra channels; version2 uses this for super-resolution.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t, dim, scale=1000.0, max_period=10000.0):
    """Sinusoidal embedding of t * scale with frequencies from 1 to 1/max_period."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t.float() * scale)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch):
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)      # -> (scale, shift)
        self.norm2 = norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = heads

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, hh * ww).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                             v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, hh, ww))


class UNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=0, ch=32, mults=(1, 2, 4, 8),
                 num_res=2, attn_res=(16, 8), dropout=0.1, image_size=64):
        super().__init__()
        emb_dim = 4 * ch
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.conv_in = nn.Conv2d(in_channels + cond_channels, ch, 3, padding=1)

        # down path: remember the channel count of every feature map we keep as a skip
        self.down = nn.ModuleList()
        skips, cur, res = [ch], ch, image_size
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blocks = [ResBlock(cur, ch * m, emb_dim, dropout)]
                cur = ch * m
                if res in attn_res:
                    blocks.append(Attention(cur))
                self.down.append(nn.ModuleList(blocks))
                skips.append(cur)
            if i != len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)]))
                skips.append(cur)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb_dim, dropout), Attention(cur),
                                  ResBlock(cur, cur, emb_dim, dropout)])

        # up path: one more block per level than the down path, each eating one skip
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                blocks = [ResBlock(cur + skips.pop(), ch * m, emb_dim, dropout)]
                cur = ch * m
                if res in attn_res:
                    blocks.append(Attention(cur))
                self.up.append(nn.ModuleList(blocks))
            if i != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(cur, cur, 3, padding=1)]))
                res *= 2

        self.out = nn.Sequential(norm(cur), nn.SiLU(), nn.Conv2d(cur, in_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return h

    def forward(self, x, t, cond=None):
        emb = self.time_mlp(timestep_embedding(t, self.ch))
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, emb)
            hs.append(h)
        h = self._run(self.mid, h, emb)
        for blocks in self.up:
            if isinstance(blocks[0], ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blocks, h, emb)
        return self.out(h)


if __name__ == '__main__':
    m = UNet()
    print(f'{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters')
    x = torch.randn(2, 3, 64, 64)
    print(m(x, torch.rand(2)).shape)
    m2 = UNet(cond_channels=3)
    print(m2(x, torch.rand(2), torch.randn(2, 3, 64, 64)).shape)
