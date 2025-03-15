import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, FashionMNIST, VisionDataset
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader
from random import sample
import matplotlib.pyplot as plt
import os
import zipfile
import json

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

    def load(self, dataset, n, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False, synth: int = None) -> tuple[DataLoader, DataLoader]:
        match dataset:
            case "mnist":
                train, test = self.load_mnist(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "fashionmnist":
                train, test = self.load_fashionmnist(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "cifar10":
                train, test = self.load_cifar10(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "cifar100":
                train, test = self.load_cifar100(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "imagenet":
                train, test = self.load_imagenet(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case "tiny-imagenet":
                train, test = self.load_tinyimagenet(n, batch_size=batch_size, out_dir=out_dir, augment=augment)
            case _:
                raise ValueError(f"Dataset {dataset} is not supported.")
        
        if synth:
            train, test = self._generate_synth(train, test, synth)

        return train, test
    
    def _generate_synth(self, train: DataLoader, test: DataLoader, m: int):
        raise(NotImplementedError)
            
    def _apply_augmentation(self, base_transform: v2.Compose, image_size: int) -> v2.Compose:
        # return v2.Compose([
        #     v2.RandomCrop(image_size, padding=4),
        #     v2.RandomHorizontalFlip(),
        #     v2.RandomRotation(15),
        #     base_transform
        # ])
        return v2.Compose([
            v2.AutoAugment(),
            v2.RandomHorizontalFlip(),
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
        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
        ])
        if augment: transform = self._apply_augmentation(transform, 28)
        trainset = MNIST(out_dir, train=True, download=True, transform=transform)
        testset = MNIST(out_dir, train=False, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_fashionmnist(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        testset = FashionMNIST(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 28)
        trainset = FashionMNIST(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar10(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = CIFAR10(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 32)
        trainset = CIFAR10(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def load_cifar100(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        testset = CIFAR100(out_dir, train=False, download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 32)
        trainset = CIFAR100(out_dir, train=True, download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def load_imagenet(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        testset = ImageNet(out_dir, split='val', download=True, transform=transform)
        if augment: transform = self._apply_augmentation(transform, 224)
        trainset = ImageNet(out_dir, split='train', download=True, transform=transform)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    def load_tinyimagenet(self, n: int = -1, batch_size: int = 128, out_dir: str = "./data/", augment: bool = False) -> tuple[DataLoader, DataLoader]:
        transform = v2.Compose([
            v2.Resize(64),
            v2.ToTensor(),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if n != -1:
            train_dataloader = self._get_n_labels(n, trainset, batch_size)
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

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

    def __init__(self, optimizer = None, warmup=False, reducer=False):
        """
        Initializes the Schedulers class with the given optimizer and warmup flag.

        Args:
            optimizer (torch.optim.Optimizer, optional): The optimizer for which to schedule the learning rate. Defaults to None.
            warmup (bool, optional): Whether to use a warmup scheduler. Defaults to False.
        """
        self.optimizer = optimizer
        self.warmup = warmup
        self.reducer = reducer

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
            case "constant+multistep": # used for resnet training
                sched = self.load_constantmultistep(**kwargs)
            case "onecycle":
                sched = self.load_onecycle(**kwargs)
            case _:
                sched = None
        
        if self.warmup:
            warmup_period = kwargs.get('warmup_period', 10)
            warmup_function = kwargs.get('warmup_function', 'linear')

            linear_lambda = lambda epoch: epoch / warmup_period
            quadratic_lambda = lambda epoch: (epoch / warmup_period) ** 2

            warmup_lambda = None
            if warmup_function == 'linear':
                warmup_lambda = linear_lambda
            elif warmup_function == 'quadratic':
                warmup_lambda = quadratic_lambda

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=warmup_lambda
            )
            
            if sched:
                sched = torch.optim.lr_scheduler.SequentialLR(
                    optimizer=self.optimizer,
                    schedulers=[warmup_scheduler, sched],
                    milestones=[warmup_period]
                )
            else:
                constant_sched = self.load_constant(
                    factor=1,
                    total_iters=1
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    optimizer=self.optimizer,
                    schedulers=[warmup_scheduler, constant_sched],
                    milestones=[warmup_period]
                )
        
        reducer = None
        if self.reducer:
            reducer = self.load_reducer(**kwargs)
        if reducer:
            return sched, reducer
        
        return sched
    
    def load_reducer(self, **kwargs):
        """
        Initializes and returns a ReduceLROnPlateau scheduler.

        Args:
            **kwargs: Arbitrary keyword arguments.
                - factor (float, optional): Factor by which the learning rate will be reduced. Default is 0.5.
                - patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Default is 5.
                - min_lr (float, optional): A lower bound on the learning rate of all param groups or each group respectively. Default is 1e-6.

        Returns:
            torch.optim.lr_scheduler.ReduceLROnPlateau: Initialized learning rate scheduler.
        """

        reducer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=kwargs.get('factor', 0.5), 
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
        return reducer
    
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
    
    def load_constantmultistep(self, **kwargs):
        """
        Creates a learning rate scheduler that first applies a constant learning rate for a specified number of epochs,
        followed by a multi-step learning rate decay.

        Parameters:
        -----------
        **kwargs : dict
            constant_epochs : int, optional
                Number of epochs to apply the constant learning rate. Default is 5.
            factor : float, optional
                Multiplicative factor of learning rate decay for the constant scheduler. Default is 0.3333333.
            milestones : list of int, optional
                List of epoch indices at which to decay the learning rate for the multi-step scheduler. Default is [30, 80].
            gamma : float, optional
                Multiplicative factor of learning rate decay for the multi-step scheduler. Default is 0.1.

        Returns:
        --------
        torch.optim.lr_scheduler.SequentialLR
            A sequential learning rate scheduler that first applies a constant learning rate and then a multi-step decay.
        """

        constant_epochs = kwargs.get('constant_epochs', 5)
        factor = kwargs.get('factor', 0.3333333)
        milestones = kwargs.get('milestones', [30, 80])
        gamma = kwargs.get('gamma', 0.1)

        constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=self.optimizer,
            factor=factor,
            total_iters=constant_epochs
        )

        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=milestones,
            gamma=gamma
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[constant_scheduler, multistep_scheduler],
            milestones=[constant_epochs]
        )

        return scheduler
    
    def load_onecycle(self, **kwargs):
        """
        Creates and returns a OneCycleLR scheduler for the optimizer.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
            - max_lr (float): Maximum learning rate. Default is 1e-3.
            - steps_per_epoch (int): Number of steps per epoch. Default is the length of the training loader.
            - total_epochs (int): Number of epochs to train. Default is 100.
            - pct_start (float): The percentage of the cycle (in number of steps) spent increasing the learning rate. Default is 0.3.

        Returns:
        torch.optim.lr_scheduler.OneCycleLR: A OneCycleLR scheduler instance.
        """
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            steps_per_epoch=kwargs.get('steps_per_epoch', kwargs.get('len_train_loader')),
            epochs=kwargs.get('total_epochs', 100),
            pct_start=kwargs.get('pct_start', 0.3)
        )
        return scheduler

def plot_metrics(train_accs, train_losses, val_accs, val_losses, plt_show=False, out="metrics.png"):
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
    else:
        plt.savefig(out)

    plt.clf()

def save_metrics(train_loss, train_acc, val_loss, val_acc, lr, prefix, epoch, best=False):
    '''
    Used for saving all metrics during training to a file.
    '''

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if os.path.exists(f"{prefix}/info.json"):
        with open(f"{prefix}/info.json", "r+") as f:
            d = json.load(f)
    else:
        d = {}

    if best: # if saving the best metrics

        d['best_res'] = {
            "epoch" : epoch+1,
            "train_loss" : train_loss,
            "train_acc" : train_acc,
            "val_loss" : val_loss,
            "val_acc" : val_acc,
        }

    else: # saving per epoch

        if "metrics" not in d:
            d['metrics'] = {}

        d['metrics'][epoch+1] = {
            "train_loss" : train_loss,
            "train_acc" : train_acc,
            "val_loss" : val_loss,
            "val_acc" : val_acc,
            "lr" : lr
        }

    with open(f"{prefix}/info.json", "w") as f:
        json.dump(d, f, indent=4)

def save_parameters(args, prefix):
    '''
    Used for saving all parameters for training to a file.
    '''

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if os.path.exists(f"{prefix}/info.json"):
        with open(f"{prefix}/info.json", "r+") as f:
            d = json.load(f)
    else:
        d = {}

    d["args"] = vars(args)

    with open(f"{prefix}/info.json", "w") as f:
        json.dump(d, f, indent=4)

def save_checkpoint(model, optimizer, lr_scheduler, epoch, filepath):
    """
    Saves the model, optimizer, and learning rate scheduler states to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler or tuple): The learning rate scheduler(s) to save.
        epoch (int): The current epoch number.
        filepath (str): The path to save the checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if isinstance(lr_scheduler, tuple):
        checkpoint['lr_scheduler_state_dict'] = [sched.state_dict() for sched in lr_scheduler if sched is not None]
    else:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    torch.save(checkpoint, f"{filepath}/epoch-{epoch}.pt")

def load_checkpoint(model, optimizer, lr_scheduler, filepath):
    """
    Loads the model, optimizer, and learning rate scheduler states from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler or tuple): The learning rate scheduler(s) to load.
        filepath (str): The path to the checkpoint file.
    
    Returns:
        int: The epoch number from the checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if isinstance(lr_scheduler, tuple):
        for sched, state_dict in zip(lr_scheduler, checkpoint['lr_scheduler_state_dict']):
            sched.load_state_dict(state_dict)
    else:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    
    return checkpoint['epoch']

if __name__ == "__main__":
    datasets = Datasets()

    def display_images(dataloader, num_images=5):
        """
        Displays a few images from the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to fetch images from.
            num_images (int, optional): The number of images to display. Defaults to 5.
        """
        images, labels = next(iter(dataloader))
        images = images[:num_images]
        labels = labels[:num_images]

        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i, (img, label) in enumerate(zip(images, labels)):
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            # img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 1, 3) + torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 1, 3)  # Unnormalize
            img = img.numpy()
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {label.item()}")
            axes[i].axis('off')
        plt.show()
        plt.savefig("cifar100.png")
        plt.close(fig)

    # Example usage 
    train_loader, _ = datasets.load("cifar100", n=-1, augment=True)
    display_images(train_loader)