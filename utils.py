import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, FashionMNIST, VisionDataset
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from random import sample
import matplotlib.pyplot as plt
import os
import zipfile

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
    - Imagenet
    - TinyImageNet
    - MNIST
    - Fashion-MNIST
    '''
    def __init__(self, seed: int = 42):
        torch.manual_seed(seed) # used for fetching random subset of data for few sample
        pass

    def load(self, dataset, n, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        match dataset:
            case "mnist":
                return self.load_mnist(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "fashionmnist":
                return self.load_fashionmnist(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "cifar10":
                return self.load_cifar10(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "cifar100":
                return self.load_cifar100(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "imagenet":
                return self.load_imagenet(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "tiny-imagenet":
                return self.load_tinyimagenet(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case _:
                raise ValueError(f"Dataset {dataset} is not supported.")
            
    def _apply_augmentation(self, base_transform: transforms.Compose, image_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            base_transform
        ])

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

    def load_mnist(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if augment: transform = self._apply_augmentation(transform, 28)
        trainset = MNIST(out_dir, train=True, download=True, transform=transform)
        testset = MNIST(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_fashionmnist(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = FashionMNIST(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 28)
        trainset = FashionMNIST(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar10(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = CIFAR10(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 32)
        trainset = CIFAR10(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar100(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        testset = CIFAR100(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 32)
        trainset = CIFAR100(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def load_imagenet(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        testset = ImageNet(out_dir, split='val', download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 224)
        trainset = ImageNet(out_dir, split='train', download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def load_tinyimagenet(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if not os.path.exists(os.path.join(out_dir, 'tiny-imagenet-200')):
            import urllib.request

            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            zip_path = os.path.join(out_dir, 'tiny-imagenet-200.zip')

            print("Downloading Tiny ImageNet dataset...")
            os.makedirs(out_dir, exist_ok=True)
            urllib.request.urlretrieve(url, zip_path)

            print("Extracting Tiny ImageNet dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(out_dir)

            val_dir = os.path.join(out_dir, 'tiny-imagenet-200', 'val')
            val_img_dir = os.path.join(val_dir, 'images')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

            with open(val_annotations_file, 'r') as f:
                val_annotations = f.readlines()

            val_dict = {}
            for line in val_annotations:
                parts = line.strip().split('\t')
                val_dict[parts[0]] = parts[1]

            for img_file, label in val_dict.items():
                label_dir = os.path.join(val_dir, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                os.rename(os.path.join(val_img_dir, img_file), os.path.join(label_dir, img_file))

            os.rmdir(val_img_dir)
            os.remove(zip_path)

        testset = torchvision.datasets.ImageFolder(root=os.path.join(out_dir, 'tiny-imagenet-200', 'val'), transform=transform)
        if augment: transform = self._apply_augmentation(transform, 64)
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(out_dir, 'tiny-imagenet-200', 'train'), transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader
    
class Schedulers():
    """
    A class to handle different learning rate schedulers with optional warmup.

    Attributes:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup (bool): Whether to use a warmup scheduler.

    Methods:
        load(scheduler, **kwargs):
            Loads the specified scheduler with given parameters.
        
        load_constant(**kwargs):
            Loads a linear learning rate scheduler with optional warmup.
            
        load_linear(**kwargs):
            Loads a linear learning rate scheduler with optional warmup.
        
        load_multistep(**kwargs):
            Loads a multi-step learning rate scheduler with optional warmup.
    """

    def __init__(self, optimizer = None, warmup=False):
        """
        Initializes the Schedulers class with the given optimizer and warmup flag.

        Args:
            optimizer (torch.optim.Optimizer, optional): The optimizer for which to schedule the learning rate. Defaults to None.
            warmup (bool, optional): Whether to use a warmup scheduler. Defaults to False.
        """
        self.optimizer = optimizer
        self.warmup = warmup

    def load(self, scheduler, **kwargs):
        """
        Loads the specified scheduler with given parameters.

        Args:
            scheduler (str): The type of scheduler to load ('constant', 'linear', or 'multistep').
            **kwargs: Additional parameters for the scheduler.
                - warmup_period (int, optional): The number of warmup iterations. Defaults to 10.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler.
        """
        match scheduler:
            case "lineardecay":
                sched = self.load_lineardecay(**kwargs)
            case "constant":
                sched = self.load_constant(**kwargs)
            case "linear":
                sched = self.load_linear(**kwargs)
            case "multistep":
                sched = self.load_multistep(**kwargs)
            case _:
                sched = None
        
        if self.warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: epoch / kwargs.get('warmup_period', 10)
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmup_scheduler, sched],
                milestones=[kwargs.get('warmup_period', 10)]
            )
            return sched
        
        return sched
    
    def load_lineardecay(self, **kwargs):
        """
        Loads a constant learning rate scheduler with optional warmup scheduler.
        Args:
            **kwargs: Arbitrary keyword arguments.
                - total_epochs (int): Number of iterations for the learning rate decay.
        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler.
        """

        total_epochs = kwargs.get('total_epochs', 30)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: 1 - x / total_epochs
        )

        return scheduler
            
    def load_constant(self, **kwargs):
        """
        Loads a constant learning rate scheduler with optional warmup scheduler.
        Args:
            **kwargs: Arbitrary keyword arguments.
                - factor (float): Multiplicative factor of learning rate decay. Default is 0.3333333.
                - total_iters (int): Number of iterations for the constant learning rate. Default is 5.
        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler.
        """
        
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=self.optimizer,
            factor=kwargs.get('factor', 0.3333333),
            total_iters=kwargs.get('total_iters', 5)
        )
        return scheduler

    def load_linear(self, **kwargs):
        """
        Loads a linear learning rate scheduler with optional warmup.

        Args:
            **kwargs: Additional parameters for the linear scheduler.
                - start_factor (float, optional): The initial learning rate factor. Defaults to 0.3333333.
                - end_factor (float, optional): The final learning rate factor. Defaults to 1.0.
                - total_iters (int, optional): The number of iterations over which to adjust the learning rate. Defaults to 5.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler.
        """

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=kwargs.get('start_factor', 0.3333333),
            end_factor=kwargs.get('end_factor', 1.0),
            total_iters=kwargs.get('total_iters', 5)
        )
        return scheduler

    def load_multistep(self, **kwargs):
        """
        Loads a multi-step learning rate scheduler with optional warmup.

        Args:
            **kwargs: Additional parameters for the multi-step scheduler.
                - milestones (list of int, optional): List of epoch indices. Must be increasing. Defaults to [30, 80].
                - gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.1.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler.
        """

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=kwargs.get('milestones', [30, 80]),
            gamma=kwargs.get('gamma', 0.1)
        )
        return scheduler

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