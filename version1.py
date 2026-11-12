"""version1: flow matching between two image distributions: super-resolution of faces.

NEW compared to version0:
  * the source x_1 is no longer noise but a degraded copy of the same face:
    x_1 = upscale(downscale(x_0)), 64x64 -> 24x24 -> 64x64 (bicubic), i.e. a blurry face.
    The training pairs (x_0, x_1) are now *coupled* (each blurry image belongs to one
    sharp image) instead of independent. The loss and the straight-line path are unchanged.
  * integrating the ODE from t = 1 to t = 0 starting at a blurry face now produces a
    sharpened face. Because the coupling is deterministic, the learned paths are nearly
    straight, so very few steps are needed: --sample-steps 1 is the one-shot prediction
    x_0 = x_1 - v(x_1, 1) (what Parag's original inference script did).

Per epoch it saves results/x_0.png (ground truth), results/x_1.png (blurry input) and
results/x_0_pred.png (the super-resolved output) for a fixed batch.

Run:  python version1.py --data-dir ./data/celeba --epochs 30
      python version1.py --sample-only --checkpoint models/version1_29.pt --sample-steps 1
"""
import torch
import torch.optim as optim
from torchvision import transforms

from common import get_args, flow_matching_loss, integrate
from dataloader import get_dataloader, save_image
from model import UNet


def get_degradation(low_res=24, high_res=64):
    down = transforms.Resize(low_res, interpolation=transforms.InterpolationMode.BICUBIC,
                             antialias=True)
    up = transforms.Resize(high_res, interpolation=transforms.InterpolationMode.BICUBIC,
                           antialias=True)

    def degrade(x):
        if x.device.type == 'mps':  # antialiased bicubic resizing is not implemented on MPS
            return up(down(x.cpu())).to(x.device)
        return up(down(x))
    return degrade


def main(args):
    degrade = get_degradation(args.low_res, 64)
    model = UNet(base=args.base).to(args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader = get_dataloader(args.data_dir, args.batch_size, shuffle=True,
                                num_workers=args.num_workers)
    # a fixed batch to visualise (the first batch of a shuffled loader)
    x_vis, _ = next(iter(get_dataloader(args.data_dir, 64, shuffle=True, num_workers=0,
                                        drop_last=False)))
    x_vis = x_vis.to(args.device)
    x_vis_1 = degrade(x_vis)

    def visualise(tag=''):
        save_image(x_vis, f'{args.out_dir}/x_0.png')
        save_image(x_vis_1, f'{args.out_dir}/x_1.png')
        save_image(integrate(model, x_vis_1, args), f'{args.out_dir}/x_0_pred{tag}.png')

    if args.sample_only:
        visualise()
        return

    for epoch in range(args.epochs):
        model.train()
        for i, (x_0, _) in enumerate(dataloader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x_0 = x_0.to(args.device)
            x_1 = degrade(x_0)  # NEW: the source is the blurry version of x_0

            loss, _, _ = flow_matching_loss(model, x_0, x_1, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch} Batch {i} Loss {loss.item():.4f}')

        visualise()
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), f'models/version1_{epoch}.pt')


if __name__ == '__main__':
    def extra(parser):
        parser.add_argument('--low-res', type=int, default=24,
                            help='side length of the degraded image')
    main(get_args('version1: flow-matching super-resolution of CelebA faces', extra=extra))
