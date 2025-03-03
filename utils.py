import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, FashionMNIST, VisionDataset
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from random import sample
import matplotlib.pyplot as plt

class Datasets():
    '''
    This is a general helper class that is meant to make it easier to load and
    handle data from multiple different datasets when training and doing
    knowledge distillation.

    Can pass in:
    - `n`: Number of images per class label
    - `batch_size`: Default `128`
    - `shuffle`: Default `True`

    Datasets includes:
    - CIFAR-10
    - CIFAR-100
    - ImageNet
    - MNIST
    - Fashion-MNIST
    '''
    def __init__(self, seed: int = 42):
        torch.manual_seed(seed) # used for fetching random subset of data for few sample
        pass

    def load(self, dataset, n, batch_size: int = 128, out_dir: str = "./data/"):
        match dataset:
            case "mnist":
                return self.load_mnist(n, batch_size=batch_size, out_dir=out_dir)
            case "fashionmnist":
                return self.load_fashionmnist(n, batch_size=batch_size, out_dir=out_dir)
            case "cifar10":
                return self.load_cifar10(n, batch_size=batch_size, out_dir=out_dir)
            case "cifar100":
                return self.load_cifar100(n, batch_size=batch_size, out_dir=out_dir)
            case "imagenet":
                return self.load_imagenet(n, batch_size=batch_size, out_dir=out_dir)
            case _:
                raise ValueError(f"Dataset {dataset} is not supported.")

    def _get_n_labels(self, n, dataset: VisionDataset, batch_size: int = 128) -> DataLoader:
        label_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        sampled_indices = []
        for label in label_to_indices.keys():
            sampled_indices += sample(label_to_indices[label], n)

        sampled_subset = Subset(dataset, sampled_indices)
        return DataLoader(sampled_subset, batch_size=batch_size, shuffle=True) 

    def load_mnist(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/") -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MNIST(out_dir, train=True, download=True, transform=transform)
        testset = MNIST(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_fashionmnist(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/") -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = FashionMNIST(out_dir, train=True, download=True, transform=transform)
        testset = FashionMNIST(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar10(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/") -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = CIFAR10(out_dir, train=True, download=True, transform=transform)
        testset = CIFAR10(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar100(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/") -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = CIFAR100(out_dir, train=True, download=True, transform=transform)
        testset = CIFAR100(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def load_imagenet(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/") -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        trainset = ImageNet(out_dir, split='train', download=True, transform=transform)
        testset = ImageNet(out_dir, split='val', download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

def plot_metrics(train_accs, train_losses, val_accs, val_losses, plt_show=True):
    '''
    Helper function for building a regular matplotlib plot.
    '''
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    color = 'tab:red'
    ax1.plot(range(len(train_losses)), train_losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot(range(len(val_losses)), val_losses, c="red", label="Val. Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(range(len(train_accs)), train_accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot(range(len(val_accs)), val_accs, c="blue", label="Val. Acc.")
    ax2.set_ylabel("Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01,1.01)
    
    fig.tight_layout()
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    if plt_show:
        plt.show()
    plt.clf()