"""version0: (conditional) flow matching: generate CelebA faces from Gaussian noise.

What this file teaches:
  * we want a velocity field v(x, t) whose ODE dx/dt = v carries noise (t = 1) to data
    (t = 0). Such an ODE defines a continuous normalizing flow; instead of training it
    by maximum likelihood (expensive), flow matching regresses v directly.
  * pick a data image x_0 and a noise image x_1 ~ N(0, I) and join them by a straight
    line x_t = (1 - t) x_0 + t x_1. Along this line the velocity is simply x_1 - x_0.
    The loss is  || v(x_t, t) - (x_1 - x_0) ||^2  at a random t ~ U(0, 1).
    Averaged over all pairs that pass through x_t, the minimiser is the velocity field
    that transports the whole noise distribution to the data distribution
    (this is the "conditional flow matching" result of Lipman et al. 2023;
    the straight-line path is also called rectified flow, Liu et al. 2023).
  * to sample: draw x_1 ~ N(0, I) and integrate the ODE from t = 1 to t = 0 with Euler steps.
  * --target residual regresses x_t - x_0 = t (x_1 - x_0) instead (an alternative Parag
    experimented with); the sampler converts it back to a velocity.

Per epoch it saves results/samples_<epoch>.png; for the first batch it also saves
results/x_t.png and results/u_t.png (the network's inputs and regression targets).

Run:  python version0.py --data-dir ./data/celeba --epochs 100
"""
import torch
import torch.optim as optim

from common import get_args, flow_matching_loss, integrate
from dataloader import get_dataloader, save_image
from model import UNet


def main(args):
    model = UNet(base=args.base).to(args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # fixed noise, so the sample grids of different epochs are comparable
    fixed_noise = torch.randn(64, 3, 64, 64, device=args.device)

    if args.sample_only:
        save_image(integrate(model, fixed_noise, args), f'{args.out_dir}/samples.png')
        return

    dataloader = get_dataloader(args.data_dir, args.batch_size, shuffle=True,
                                num_workers=args.num_workers)
    for epoch in range(args.epochs):
        model.train()
        for i, (x_0, _) in enumerate(dataloader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x_0 = x_0.to(args.device)
            x_1 = torch.randn_like(x_0)  # the source distribution: pure noise

            loss, x_t, u_t = flow_matching_loss(model, x_0, x_1, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == 0 and i == 0:
                save_image(x_t[:32], f'{args.out_dir}/x_t.png')
                save_image(u_t[:32], f'{args.out_dir}/u_t.png')
            if i % 100 == 0:
                print(f'Epoch {epoch} Batch {i} Loss {loss.item():.4f}')

        save_image(integrate(model, fixed_noise, args), f'{args.out_dir}/samples_{epoch}.png')
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), f'models/version0_{epoch}.pt')


if __name__ == '__main__':
    main(get_args('version0: flow matching from noise to CelebA faces'))
