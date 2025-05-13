# data/mnist_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=128):
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
