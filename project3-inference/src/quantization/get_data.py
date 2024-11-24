import numpy as np
import copy
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
torch.manual_seed(42)
train_size = 45000
val_size = 5000
num_workers = 2 if torch.cuda.is_available() else 0


def get_dataloaders(BATCH_SIZE):
    print("Getting dataloaders")
    train_dataset = CIFAR10(
        root="../../data/", train=True, download=False, transform=transform
    )
    test_dataset = CIFAR10(root="../../data/", train=False, transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader
