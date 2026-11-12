"""CelebA loading (torchvision ImageFolder) and image saving helpers."""

import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_dataset_statistics():
    """Per-channel mean / std of CelebA at 64x64 (recompute with: python dataloader.py --stats)."""
    return (0.5081, 0.4220, 0.3765), (0.3012, 0.2787, 0.2773)


def get_dataset(path, image_size=64):
    mean, std = get_dataset_statistics()
    return datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]))


def get_dataloader(path, batch_size, shuffle=True, num_workers=4, image_size=64, drop_last=True):
    dataset = get_dataset(path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last)


def save_image(x, name):
    """Undo the normalisation and save a grid of images."""
    mu, sigma = get_dataset_statistics()
    mu = torch.tensor(mu, device=x.device).view(1, 3, 1, 1)
    sigma = torch.tensor(sigma, device=x.device).view(1, 3, 1, 1)
    x = (x * sigma + mu).clamp(0, 1)
    vutils.save_image(vutils.make_grid(x), name)


if __name__ == "__main__":
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='Recompute the dataset statistics')
    parser.add_argument('--data-dir', type=str, default='./data/celeba')
    args = parser.parse_args()

    # statistics of the *unnormalised* images
    dataset = datasets.ImageFolder(args.data_dir, transform=transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=512, num_workers=4)
    mu, std = 0, 0
    for x, _ in tqdm(loader):
        mu += x.mean(dim=(0, 2, 3)) * len(x)
        std += x.std(dim=(0, 2, 3)) * len(x)
    print('mean:', mu / len(dataset), 'std:', std / len(dataset))
