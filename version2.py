"""version2: super-resolution (24x24 -> 64x64) as a *conditional* flow.

NEW compared to version1: the velocity field also sees the low-resolution face,
    v_theta(x_t, t | y),   y = bicubic upsampling of the 24x24 image back to 64x64,
fed to the U-Net as 3 extra input channels. Everything else (noise source, straight
paths, target x_1 - x_0, EMA, samplers) is the same as version1.

Why not flow straight from the blurry image to the sharp one (what this repo did before)?
Many sharp faces share one 24x24 version, so from a given blurry image the regression
target is the *average* of all of them: E[x_1 | blurry]. A deterministic flow started at
the blurry image, especially with one step, outputs that average, i.e. a blurry face.
Starting from noise and conditioning on y lets the flow pick ONE plausible sharp face
per noise sample. (In theory the one-step Euler output from noise is again the average,
E[x_1 | y]; the grid in the README shows it next to real samples.)

Run:
  python version2.py --data-dir /path/to/img_align_celeba --minutes 44
  python version2.py --sample-only --checkpoint checkpoints/version2.pt
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from celeba import get_loader, save_grid, FlatImageFolder
from flows import cfm_loss, integrate, as_velocity, EMA, NFE_PER_STEP
from unet import UNet
from version1 import get_device


def get_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data-dir', default='./data/celeba/img_align_celeba')
    p.add_argument('--low-res', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-steps', type=int, default=200_000)
    p.add_argument('--minutes', type=float, default=44)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--sample-every', type=int, default=2000)
    p.add_argument('--method', default='euler', choices=list(NFE_PER_STEP))
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--checkpoint', default='checkpoints/version2.pt')
    p.add_argument('--sample-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='continue training from --checkpoint')
    p.add_argument('--out-dir', default='results/version2')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def degrade(x, low_res, size=64):
    """64x64 -> low_res x low_res (antialiased bicubic): the observed low-res face."""
    dev = x.device
    if dev.type == 'mps':          # antialiased bicubic is not implemented on MPS
        x = x.cpu()
    lo = F.interpolate(x, size=low_res, mode='bicubic', antialias=True, align_corners=False)
    return lo.clamp(-1, 1).to(dev)


def upsample(lo, size=64):
    """The conditioning image y: bicubic upsampling of the low-res face back to 64x64."""
    return F.interpolate(lo, size=size, mode='bicubic', align_corners=False).clamp(-1, 1)


@torch.no_grad()
def super_resolve(model, y, method='euler', steps=50, noise=None):
    model.eval()
    x0 = torch.randn_like(y) if noise is None else noise
    return integrate(as_velocity(model, cond=y), x0, steps=steps, method=method).clamp(-1, 1)


def psnr(a, b):
    """PSNR in dB for images in [-1, 1] (peak-to-peak range 2), per image."""
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(4.0 / mse)


def main(args):
    device = get_device()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    model = UNet(cond_channels=3, ch=args.ch).to(device)
    if device.type == 'cuda':                       # faster convolutions on GPU
        model = model.to(memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay)
    print(f'U-Net: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, device {device}')

    # a fixed batch of *held-out* (test split) faces to visualise
    test = FlatImageFolder(args.data_dir, split='test', flip=False)
    x_vis = torch.stack([test[i] for i in range(16)]).to(device)
    y_vis = upsample(degrade(x_vis, args.low_res))
    noise_vis = torch.randn(x_vis.shape, generator=torch.Generator().manual_seed(1)).to(device)

    def visualise(name):
        out = super_resolve(ema.model, y_vis, args.method, args.steps, noise_vis)
        save_grid(torch.cat([y_vis[:8], out[:8], x_vis[:8]]), f'{args.out_dir}/{name}')

    if args.sample_only:
        ema.model.load_state_dict(torch.load(args.checkpoint, map_location=device)['ema'])
        visualise(f'sr_{args.method}{args.steps}.png')
        return

    loader = get_loader(args.data_dir, args.batch_size, args.num_workers, split='train')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')

    step, start, running = 0, time.time(), 0.0
    if args.resume:
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck['model'])
        ema.model.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        step = ema.num_updates = ck['step']
        sched.last_epoch = step
        print(f'resumed from step {step}')
    log = open(f'{args.out_dir}/loss.csv', 'a' if args.resume else 'w')
    while step < args.max_steps and time.time() - start < 60 * args.minutes:
        for x1 in loader:
            x1 = x1.to(device, non_blocking=True)
            y = upsample(degrade(x1, args.low_res))     # NEW: the condition
            x0 = torch.randn_like(x1)
            model.train()
            with torch.autocast(**amp):
                loss = cfm_loss(model, x0, x1, cond=y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            running += loss.item()
            if step % 100 == 0:
                print(f'step {step:6d}  loss {running / 100:.4f}  {(time.time() - start) / 60:.1f} min', flush=True)
                log.write(f'{step},{running / 100:.5f}\n')
                log.flush()
                running = 0.0
            if step % args.sample_every == 0:
                visualise(f'sr_{step:06d}.jpg')
            if step >= args.max_steps or time.time() - start > 60 * args.minutes:
                break

    torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                'args': vars(args)}, args.checkpoint)
    visualise(f'sr_final_{args.method}{args.steps}.png')
    print(f'done: {step} steps in {(time.time() - start) / 60:.1f} min -> {args.checkpoint}')


if __name__ == '__main__':
    main(get_args())
