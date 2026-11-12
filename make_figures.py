"""Make the README figures and numbers from trained version1 / version2 checkpoints.

  python make_figures.py --data-dir /path/to/img_align_celeba \
      --v1 checkpoints/version1.pt --v2 checkpoints/version2.pt --out-dir static

version1: a 8x8 grid of samples, a solver / step-count comparison on the same noise, and a
          "regression to the mean" check: the per-pixel standard deviation across generated
          faces compared with the same statistic over real (held-out) faces. A model that
          regresses to the mean face has a ratio far below 1.
version2: low-res input / bicubic / 1-step output / two samples / ground truth for held-out
          faces, and PSNRs on 512 held-out faces.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from celeba import FlatImageFolder, save_grid, to_unit
from flows import integrate, as_velocity, NFE_PER_STEP
from unet import UNet
from version1 import get_device
from version2 import degrade, upsample, super_resolve, psnr


def load_ema(path, device, **kw):
    ck = torch.load(path, map_location=device)
    a = ck['args']
    model = UNet(ch=a['ch'], **kw).to(device)
    model.load_state_dict(ck['ema'])
    return model.eval(), ck['step']


def test_images(root, n, device):
    ds = FlatImageFolder(root, split='test', flip=False)
    return torch.stack([ds[i] for i in range(n)]).to(device)


def labelled_rows(rows, labels, path, title=None, label_width=2.7):
    """rows: list of (n, 3, 64, 64) tensors in [-1, 1]; one labelled image row each."""
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), 1, figsize=(label_width + 0.72 * n, 0.95 * len(rows) + 0.5))
    for ax, r, lab in zip(axes, rows, labels):
        ax.imshow(make_grid(to_unit(r.float().cpu()), nrow=n, padding=1, pad_value=1).permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, pil_kwargs={'quality': 88})
    plt.close(fig)
    print('saved', path, f'{os.path.getsize(path) / 1e3:.0f} kB')


@torch.no_grad()
def figures_v1(args, device):
    model, step = load_ema(args.v1, device)
    v = as_velocity(model)
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(64, 3, 64, 64, generator=g).to(device)
    x = integrate(v, noise, steps=50, method='heun').clamp(-1, 1)
    save_grid(x, f'{args.out_dir}/version1_samples.jpg')

    settings = [('euler', 1), ('euler', 2), ('euler', 5), ('euler', 10), ('euler', 100),
                ('midpoint', 5), ('heun', 5), ('rk4', 5), ('heun', 50)]
    rows, labels = [], []
    for method, steps in settings:
        rows.append(integrate(v, noise[:10], steps=steps, method=method).clamp(-1, 1))
        labels.append(f'{method} {steps} step{"s" if steps > 1 else ""}\n(NFE {steps * NFE_PER_STEP[method]})')
    labelled_rows(rows, labels, f'{args.out_dir}/version1_solvers.jpg',
                  f'version1 after {step} training steps: same noise, different ODE solvers')

    # regression-to-the-mean check
    real = test_images(args.data_dir, 512, device)
    fake = []
    for i in range(0, 512, 128):
        z = torch.randn(128, 3, 64, 64, generator=g).to(device)
        fake.append(integrate(v, z, steps=50, method='heun').clamp(-1, 1))
    fake = torch.cat(fake)
    one_step = integrate(v, torch.randn(512, 3, 64, 64, generator=g).to(device), steps=1).clamp(-1, 1)
    std_real = real.std(0).mean().item()
    lines = [f'version1 ({step} steps): per-pixel std across 512 images, averaged over pixels',
             f'  real held-out faces   {std_real:.3f}',
             f'  heun 50 steps         {fake.std(0).mean().item():.3f}  (ratio {fake.std(0).mean().item() / std_real:.2f})',
             f'  euler 1 step          {one_step.std(0).mean().item():.3f}  (ratio {one_step.std(0).mean().item() / std_real:.2f})',
             f'  distance of the 1-step output to the mean real face: '
             f'{(one_step.mean(0) - real.mean(0)).abs().mean().item():.3f} (mean abs, [-1, 1] scale)']
    return lines


@torch.no_grad()
def figures_v2(args, device):
    model, step = load_ema(args.v2, device, cond_channels=3)
    x = test_images(args.data_dir, 512, device)
    lo = degrade(x, args.low_res)
    y = upsample(lo)
    g = torch.Generator().manual_seed(3)
    n = 8
    nearest = F.interpolate(lo[:n], size=64, mode='nearest')
    one = super_resolve(model, y[:n], 'euler', 1, torch.randn(n, 3, 64, 64, generator=g).to(device))
    sa = super_resolve(model, y[:n], 'heun', 25, torch.randn(n, 3, 64, 64, generator=g).to(device))
    sb = super_resolve(model, y[:n], 'heun', 25, torch.randn(n, 3, 64, 64, generator=g).to(device))
    labelled_rows([nearest, y[:n], one, sa, sb, x[:n]],
                  [f'input {args.low_res}x{args.low_res}', 'bicubic', 'flow, 1 step\n(~ the mean)',
                   'flow, heun 25\nsample A', 'flow, heun 25\nsample B', 'ground truth'],
                  f'{args.out_dir}/version2_superres.jpg',
                  f'version2 after {step} training steps, held-out faces')

    res = {'bicubic': y}
    res['flow, euler 1 step'] = torch.cat([super_resolve(model, y[i:i + 128], 'euler', 1) for i in range(0, 512, 128)])
    res['flow, heun 25 steps'] = torch.cat([super_resolve(model, y[i:i + 128], 'heun', 25) for i in range(0, 512, 128)])
    lines = [f'version2 ({step} steps), 512 held-out faces: PSNR to ground truth | PSNR of the '
             f'output downsampled to {args.low_res}px vs the input | sharpness (mean |Laplacian|)']
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sharp = lambda im: F.conv2d(im, lap, groups=3).abs().mean().item()
    lines.append(f'  {"ground truth":22s}      -   |      -   | {sharp(x):.3f}')
    for name, out in res.items():
        lines.append(f'  {name:22s} {psnr(out, x).mean().item():6.2f} | '
                     f'{psnr(degrade(out, args.low_res), lo).mean().item():6.2f} | {sharp(out):.3f}')
    return lines


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--v1', default=None)
    p.add_argument('--v2', default=None)
    p.add_argument('--low-res', type=int, default=24)
    p.add_argument('--out-dir', default='static')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    torch.manual_seed(0)
    lines = []
    if args.v1:
        lines += figures_v1(args, device)
    if args.v2:
        lines += figures_v2(args, device)
    print('\n'.join(lines))
