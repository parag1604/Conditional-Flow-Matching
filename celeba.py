"""CelebA from a flat folder of jpgs (e.g. the unzipped img_align_celeba/), and image saving.

Images are centre-cropped to 140x140 (the face), resized to 64x64, randomly flipped
and scaled to [-1, 1], the same range as the N(0, I) noise we start from.
"""
import os

import torch
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EXTENSIONS = ('.jpg', '.jpeg', '.png')


class FlatImageFolder(Dataset):
    """All images directly inside `root` (no class sub-folders, unlike ImageFolder)."""
    def __init__(self, root, split='all', image_size=64, crop=140, flip=True):
        files = sorted(os.path.join(root, f) for f in os.listdir(root)
                       if f.lower().endswith(EXTENSIONS))
        if not files:
            raise FileNotFoundError(f'no images found in {root}')
        # official CelebA partition: 1-162770 train, 162771-182637 val, 182638-202599 test;
        # for any other folder the last 10% of the sorted files are held out as 'test'
        n_train = 162770 if len(files) == 202599 else int(0.9 * len(files))
        n_test = 182637 if len(files) == 202599 else n_train
        self.files = {'all': files, 'train': files[:n_train], 'test': files[n_test:]}[split]
        tf = [transforms.CenterCrop(crop), transforms.Resize(image_size, antialias=True)]
        if flip:
            tf.append(transforms.RandomHorizontalFlip())
        tf += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            return self.transform(im.convert('RGB'))


def get_loader(root, batch_size, num_workers=8, shuffle=True, flip=True, split='train'):
    return DataLoader(FlatImageFolder(root, split=split, flip=flip), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True, pin_memory=True,
                      persistent_workers=num_workers > 0)


def to_unit(x):
    """[-1, 1] -> [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


def save_grid(x, path, nrow=8, padding=2):
    """Save a batch of [-1, 1] images as a grid (jpg keeps the files small)."""
    grid = vutils.make_grid(to_unit(x.float().cpu()), nrow=nrow, padding=padding, pad_value=1.0)
    img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
    if path.lower().endswith(('.jpg', '.jpeg')):
        img.save(path, quality=90)
    else:
        img.save(path)
