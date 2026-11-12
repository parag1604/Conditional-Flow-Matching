"""version3: minibatch optimal-transport coupling (2-D toy), for straighter flows.

NEW compared to version0: inside every minibatch, the noise samples x_0 and the data
samples x_1 are no longer paired at random. We re-pair them with the permutation that
minimises the total squared distance sum_i ||x_0^(i) - x_1^(pi(i))||^2 (an assignment
problem, solved exactly by the Hungarian algorithm, scipy's linear_sum_assignment).
Loss and sampler are unchanged.

Why: with independent pairs, straight lines from different pairs cross; the network can
only learn their average at a crossing, so the learned trajectories bend, and few-step
Euler sampling is inaccurate (1 Euler step gives the data mean). With OT pairs the lines
rarely cross, so the learned flow is close to straight and a few steps suffice
(Tong et al. 2024; Pooladian et al. 2023). In high dimension (images) with small batches
the effect is much weaker, which is why the image versions keep independent pairs.

Run:  python version3.py --data moons
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment

from flows import integrate, as_velocity
from version0 import get_parser, train, sample_data


def ot_coupling(x0, x1):
    cost = torch.cdist(x0, x1) ** 2
    rows, cols = linear_sum_assignment(cost.numpy())
    return x0[rows], x1[cols]


def path_straightness(model, x0, steps=100):
    """Mean over samples of  (length of the ODE path) / (distance from start to end)  - 1:
    0 for perfectly straight trajectories."""
    path = integrate(as_velocity(model), x0, steps=steps, method='euler', return_path=True)
    length = (path[1:] - path[:-1]).norm(dim=-1).sum(0)
    chord = (path[-1] - path[0]).norm(dim=-1)
    return (length / chord - 1).mean().item()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    models = {'independent pairs (version0)': train(args),
              'minibatch-OT pairs (version3)': train(args, coupling=ot_coupling)}
    torch.manual_seed(123)
    x0 = torch.randn(2000, 2)
    data = sample_data(args.data, 2000)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for row, (name, model) in zip(axes, models.items()):
        v = as_velocity(model)
        path = integrate(v, x0[:300], steps=100, method='euler', return_path=True)
        ax = row[0]
        for i in range(300):
            ax.plot(path[:, i, 0], path[:, i, 1], color='0.6', lw=0.4, alpha=0.6)
        ax.scatter(path[0, :, 0], path[0, :, 1], s=3, color='k', zorder=3)
        ax.scatter(path[-1, :, 0], path[-1, :, 1], s=3, color='C0', zorder=3)
        s = path_straightness(model, x0[:1000])
        ax.set_title(f'{name}\npath length / chord - 1 = {s:.3f}', fontsize=9)
        for ax, steps in zip(row[1:], (1, 2, 5)):
            x = integrate(v, x0, steps=steps, method='euler')
            ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.15, color='C1')
            ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5, color='C0')
            ax.set_title(f'Euler {steps} step{"s" if steps > 1 else ""} (blue) vs data (orange)', fontsize=9)
        print(f'{name}: straightness {s:.3f}')
    for ax in axes.flat:
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal')
        ax.set_xlabel('$x^{(1)}$')
        ax.set_ylabel('$x^{(2)}$')
    fig.tight_layout()
    path = f'{args.out_dir}/{args.data}_ot.png'
    fig.savefig(path, dpi=80)
    print('saved', path)


if __name__ == '__main__':
    main(get_parser().parse_args())
