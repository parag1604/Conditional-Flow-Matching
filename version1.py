"""version1: unconditional face generation on CelebA 64x64 with (rectified) flow matching.

NEW compared to version0 (the 2-D toy): the same loss and samplers, but
  * the data are 64x64x3 images, so v_theta is a U-Net (unet.py) instead of an MLP;
  * an exponential moving average (EMA) of the weights is kept and used for sampling;
  * mixed precision (bf16) on CUDA, and a time budget so runs stay short.

Training:   x_0 ~ N(0, I),  x_1 = a face,  t ~ U(0, 1),  x_t = (1 - t) x_0 + t x_1
            loss = || v_theta(x_t, t) - (x_1 - x_0) ||^2
Sampling:   draw x_0 ~ N(0, I) and integrate dx/dt = v_theta(x, t) from t = 0 to t = 1
            with Euler / midpoint / Heun / RK4 (`--method`, `--steps`).

Run:
  python version1.py --data-dir /path/to/img_align_celeba --minutes 44
  python version1.py --sample-only --checkpoint checkpoints/version1.pt --method heun --steps 25
"""
import argparse
import os
import time

import torch

from celeba import get_loader, save_grid
from flows import cfm_loss, integrate, as_velocity, EMA, NFE_PER_STEP
from unet import UNet


def get_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data-dir', default='./data/celeba/img_align_celeba')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-steps', type=int, default=200_000)
    p.add_argument('--minutes', type=float, default=44, help='stop training after this time')
    p.add_argument('--ch', type=int, default=32, help='U-Net base width')
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--sample-every', type=int, default=2000)
    p.add_argument('--method', default='euler', choices=list(NFE_PER_STEP))
    p.add_argument('--steps', type=int, default=50, help='ODE steps when sampling')
    p.add_argument('--checkpoint', default='checkpoints/version1.pt')
    p.add_argument('--sample-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='continue training from --checkpoint')
    p.add_argument('--out-dir', default='results/version1')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def generate(model, noise, method, steps):
    model.eval()
    return integrate(as_velocity(model), noise, steps=steps, method=method).clamp(-1, 1)


def main(args):
    device = get_device()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    model = UNet(ch=args.ch).to(device)
    if device.type == 'cuda':                       # faster convolutions on GPU
        model = model.to(memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay)
    print(f'U-Net: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, device {device}')
    fixed_noise = torch.randn(64, 3, 64, 64, generator=torch.Generator().manual_seed(1)).to(device)

    if args.sample_only:
        ema.model.load_state_dict(torch.load(args.checkpoint, map_location=device)['ema'])
        x = generate(ema.model, fixed_noise, args.method, args.steps)
        save_grid(x, f'{args.out_dir}/samples_{args.method}{args.steps}.png')
        return

    loader = get_loader(args.data_dir, args.batch_size, args.num_workers, split='train')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
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
            x1 = x1.to(device, non_blocking=True)      # data   (t = 1)
            x0 = torch.randn_like(x1)                   # noise  (t = 0)
            model.train()
            with torch.autocast(**amp):
                loss = cfm_loss(model, x0, x1)
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
                save_grid(generate(ema.model, fixed_noise, 'euler', 50), f'{args.out_dir}/samples_{step:06d}.jpg')
            if step >= args.max_steps or time.time() - start > 60 * args.minutes:
                break

    torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                'args': vars(args)}, args.checkpoint)
    save_grid(generate(ema.model, fixed_noise, args.method, args.steps),
              f'{args.out_dir}/samples_final_{args.method}{args.steps}.png')
    print(f'done: {step} steps in {(time.time() - start) / 60:.1f} min -> {args.checkpoint}')


if __name__ == '__main__':
    main(get_args())
