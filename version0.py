"""version0: flow matching in 2-D, from a Gaussian to two moons (or 8 Gaussians).

The whole method in a few lines:
  * x_0 ~ N(0, I) (noise, t = 0) and x_1 ~ p_data (t = 1), drawn independently;
  * the straight line x_t = (1 - t) x_0 + t x_1 has velocity x_1 - x_0;
  * train an MLP v_theta(x, t) with  loss = || v_theta(x_t, t) - (x_1 - x_0) ||^2,  t ~ U(0, 1);
  * sample by integrating dx/dt = v_theta(x, t) from t = 0 to t = 1 (flows.integrate).

The minimiser of that loss is v*(x, t) = E[x_1 - x_0 | x_t = x], the *marginal* velocity
field, and its ODE carries N(0, I) exactly onto p_data (Lipman et al. 2023; Liu et al.
2023). Note that it depends on t: the same point x means "mostly noise, keep going" at
t = 0.1 and "almost a data point, stop" at t = 0.9.

`--ignore-time` feeds t = 0.5 to the network always. The best it can then learn is the
time-averaged velocity, and the samples collapse towards the middle of the data:
the "regression to the mean" failure of the earlier image code, in 2-D.

Run:  python version0.py --data moons
      python version0.py --data 8gaussians
      python version0.py --data moons --ignore-time
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from flows import cfm_loss, integrate, as_velocity


def sample_data(name, n):
    if name == 'moons':
        from sklearn.datasets import make_moons
        x, _ = make_moons(n, noise=0.05)
        x = (x - np.array([0.5, 0.25])) * 2.0          # centre and scale to about [-3, 3]
        return torch.tensor(x, dtype=torch.float32)
    if name == '8gaussians':
        angles = torch.randint(0, 8, (n,)) * (2 * math.pi / 8)
        centres = 3.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return centres + 0.15 * torch.randn(n, 2)
    raise ValueError(name)


class MLP(nn.Module):
    """v_theta(x, t): t enters through a small sinusoidal embedding."""
    def __init__(self, hidden=256, n_freq=8, ignore_time=False):
        super().__init__()
        self.ignore_time = ignore_time
        self.register_buffer('freqs', 2.0 ** torch.arange(n_freq) * math.pi)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * n_freq, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, t):
        if self.ignore_time:
            t = torch.full_like(t, 0.5)
        a = t[:, None] * self.freqs[None]
        return self.net(torch.cat([x, torch.sin(a), torch.cos(a)], dim=1))


def train(args, coupling=None):
    torch.manual_seed(args.seed)
    model = MLP(ignore_time=args.ignore_time)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(args.iters):
        x1 = sample_data(args.data, args.batch_size)
        x0 = torch.randn_like(x1)
        if coupling is not None:              # used by version3 (minibatch OT)
            x0, x1 = coupling(x0, x1)
        loss = cfm_loss(model, x0, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 1000 == 0:
            print(f'iter {it + 1:5d}  loss {loss.item():.4f}')
    return model.eval()


def plot_results(model, args, path, title):
    """Top: samples after 1, 2, 5 and 100 Euler steps, and the data.
    Bottom: trajectories from noise to data, and the learned field at three times."""
    torch.manual_seed(123)
    x0 = torch.randn(2000, 2)
    data = sample_data(args.data, 2000)
    v = as_velocity(model)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for ax, steps in zip(axes[0, :3], (1, 5, 100)):
        x = integrate(v, x0, steps=steps, method='euler')
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5, color='C0')
        ax.set_title(f'samples, Euler {steps} step{"s" if steps > 1 else ""}')
    axes[0, 3].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, color='C1')
    axes[0, 3].set_title('data')

    path_ = integrate(v, x0[:300], steps=100, method='euler', return_path=True)
    ax = axes[1, 0]
    for i in range(300):
        ax.plot(path_[:, i, 0], path_[:, i, 1], color='0.6', lw=0.4, alpha=0.6)
    ax.scatter(path_[0, :, 0], path_[0, :, 1], s=3, color='k', label='t = 0 (noise)', zorder=3)
    ax.scatter(path_[-1, :, 0], path_[-1, :, 1], s=3, color='C0', label='t = 1 (samples)', zorder=3)
    ax.legend(loc='lower left', fontsize=7)
    ax.set_title('trajectories (100 Euler steps)')
    g = torch.linspace(-4, 4, 21)
    gx, gy = torch.meshgrid(g, g, indexing='xy')
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    with torch.no_grad():
        for ax, t in zip(axes[1, 1:], (0.0, 0.5, 0.9)):
            u = v(grid, t)
            ax.quiver(grid[:, 0], grid[:, 1], u[:, 0], u[:, 1], color='C2', angles='xy',
                      scale_units='xy', scale=8, width=0.004)
            ax.set_title(f'learned velocity v(x, t = {t})')
    for ax in axes.flat:
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal')
        ax.set_xlabel('$x^{(1)}$')
        ax.set_ylabel('$x^{(2)}$')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print('saved', path)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', default='moons', choices=['moons', '8gaussians'])
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--ignore-time', action='store_true',
                   help='hide t from the network (shows the collapse to the mean)')
    p.add_argument('--out-dir', default='results/version0')
    p.add_argument('--seed', type=int, default=0)
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = train(args)
    tag = f'{args.data}{"_no_time" if args.ignore_time else ""}'
    plot_results(model, args, f'{args.out_dir}/{tag}.png',
                 f'flow matching, N(0, I) -> {args.data}' + (' (t hidden from the network)' if args.ignore_time else ''))
