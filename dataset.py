import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import transforms
from tqdm import tqdm, trange


URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'


def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)


# Apparently torchvision doesn't have any loader for this so I made one
# Advantage compared to without loader: get "for free" transforms, DataLoader
# (workers), etc
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1


def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in tqdm(train_dl, "Compute normalization"):
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()


def get_train_loader(batch_size, workers):
    dataset = EuroSAT(
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    )

    trainval, test_ds = random_split(dataset, 0.9, random_state=42)
    train_ds, val_ds = random_split(trainval, 0.9, random_state=7)

    # load train dataset with computed normalization
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    mean, std = calc_normalization(train_dl)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))
    normalization = {'mean': mean, 'std': std}

    # load val dataset
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
    )

    return train_dl, val_dl, normalization, dataset.classes


def get_test_loader(normalization, batch_size, workers):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
    dataset = EuroSAT(transform=tr)
    _, test = random_split(dataset, 0.9, random_state=42)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return test_dl, dataset.classes
