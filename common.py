"""Pieces shared by version0 and version1: CLI, device, the flow-matching loss and the ODE sampler."""

import os
import argparse

import torch
import torch.nn.functional as F


def get_args(description, extra=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=str, default='./data/celeba',
                        help='ImageFolder root, e.g. ./data/celeba/img_align_celeba/*.jpg')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target', choices=['velocity', 'residual'], default='velocity',
                        help="regression target: x_1 - x_0 ('velocity') or x_t - x_0 ('residual')")
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='the network sees t * time-scale (try 1000)')
    parser.add_argument('--sample-steps', type=int, default=50,
                        help='Euler steps when integrating the ODE from t=1 to t=0')
    parser.add_argument('--base', type=int, default=64,
                        help='UNet width (64 = original model, ~62M parameters)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='stop each epoch after this many batches (quick runs)')
    parser.add_argument('--save-every', type=int, default=10, help='epochs between checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='load these weights before training / sampling')
    parser.add_argument('--sample-only', action='store_true',
                        help='skip training: load --checkpoint and only sample')
    parser.add_argument('--out-dir', type=str, default='./results')
    if extra is not None:
        extra(parser)
    args = parser.parse_args()
    args.device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    return args


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def interpolate(x_0, x_1):
    """Draw t ~ U(0, 1) per image and return the point x_t on the straight line from
    x_0 (data, t=0) to x_1 (source, t=1), together with t."""
    t = torch.rand((x_0.shape[0], 1, 1, 1), dtype=x_0.dtype, device=x_0.device)
    x_t = x_0 * (1 - t) + x_1 * t
    return x_t, t


def regression_target(x_0, x_1, x_t, target):
    """The conditional velocity of the path x_t = (1 - t) x_0 + t x_1 is dx_t/dt = x_1 - x_0.
    'residual' instead regresses x_t - x_0 = t (x_1 - x_0); from it one recovers
    x_0 = x_t - prediction directly, and the velocity as prediction / t."""
    return x_1 - x_0 if target == 'velocity' else x_t - x_0


def flow_matching_loss(model, x_0, x_1, args):
    x_t, t = interpolate(x_0, x_1)
    u_t = regression_target(x_0, x_1, x_t, args.target)
    v_t = model(x_t, t.view(-1) * args.time_scale)
    return F.mse_loss(v_t, u_t), x_t, u_t


@torch.no_grad()
def integrate(model, x_1, args, steps=None):
    """Euler integration of dx/dt = v(x, t) from t = 1 (source) back to t = 0 (data).

    With a single step this is x_0 = x_1 - v(x_1, 1): one jump along the predicted
    straight line. More steps let the trajectory bend, correcting the errors of the
    individual predictions.
    """
    steps = steps or args.sample_steps
    model.eval()
    x = x_1.clone()
    ts = torch.linspace(1, 0, steps + 1, device=x.device)
    for i in range(steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        t_batch = torch.full((x.shape[0],), float(t), device=x.device)
        pred = model(x, t_batch * args.time_scale)
        velocity = pred if args.target == 'velocity' else pred / t
        x = x - dt * velocity
    return x
