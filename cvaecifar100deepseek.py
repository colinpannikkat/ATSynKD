import argparse
import datetime
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from scipy.stats import norm
from tqdm import tqdm
from utils import Datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="0", type=str, help='gpu id')
parser.add_argument('--dataset', default="cifar100", type=str, help='dataset name')
parser.add_argument('--cvaedatatype', default="subset", type=str, help='dataset type to train cvae')
parser.add_argument('--cvaemethod', default="input_cond", type=str, help='method to train cvae')
parser.add_argument('--teacherdatatype', default="original", type=str, help='dataset type to train teacher')
parser.add_argument('--latent', default=2, type=int, help='latent dimension')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
args = parser.parse_args()

print(f"""Training Parameters:
    GPU: {args.gpu}
    Dataset: {args.dataset}
    CVAE Data Type: {args.cvaedatatype}
    CVAE Method: {args.cvaemethod}
    Teacher Data Type: {args.teacherdatatype}
    Latent Dimension: {args.latent}
    Batch Size: {args.batchsize}
    Epochs: {args.epochs}""")

# Set device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(123)
np.random.seed(123)

# Hyperparameters
img_shape = None
n_class = None
n_feature = None

# train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
train_loader, test_loader = (None, None)

if args.dataset == "cifar100":
    train_loader, test_loader = Datasets().load_cifar100(batch_size=args.batchsize, n=50, normalize=False)
    img_shape = (3, 32, 32)
    img_size = 32
    n_class = 100
    n_feature = 32 * 32 * 3
elif args.dataset == "tiny-imagenet":
    train_loader, test_loader = Datasets().load_tinyimagenet(batch_size=args.batchsize, n=50, normalize=False)
    img_shape = (3, 64, 64)
    img_size = 64
    n_class = 200
    n_feature = 64 * 64 * 3
else:
    raise ValueError("Dataset not supported")

print(f"Image shape: {img_shape}, Number of classes: {n_class}, Number of features: {n_feature}")

# Define CVAE model
class Encoder(nn.Module):
    def __init__(self, latent_dim, n_class, method):
        super(Encoder, self).__init__()
        self.method = method
        self.n_class = n_class
        
        if method == "input_cond":
            self.cond_proj = nn.Sequential(
                nn.Linear(n_class, img_size * img_size),
                nn.ReLU(),
                nn.Unflatten(1, (1, img_size, img_size))
            )
            self.conv1 = nn.Conv2d(4, 32, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*1*1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU())
        
        if method == "feature_cond":
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*1*1 + n_class, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU())
        
        self.mu = nn.Linear(16, latent_dim)
        self.log_var = nn.Linear(16, latent_dim)
    
    def forward(self, x, cond):
        if self.method == "input_cond":
            cond_proj = self.cond_proj(cond)
            x = torch.cat([x, cond_proj], dim=1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        if img_size == 64:
            x = F.relu(self.conv6(x))
        
        if self.method == "feature_cond":
            x = torch.cat([x.view(x.size(0), -1), cond], dim=1)
        
        x = self.fc(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_class):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_class = n_class
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + n_class, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 512*1*1),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)))
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid())
    
    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        x = self.fc(z)
        x = self.deconv(x)
        return x.view(-1, n_feature)

class CVAE(nn.Module):
    def __init__(self, latent_dim, n_class, method):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim, n_class, method)
        self.decoder = Decoder(latent_dim, n_class)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, cond):
        mu, log_var = self.encoder(x, cond)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z, cond)
        return recon, mu, log_var
    
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

# Initialize model
cvae = CVAE(args.latent, n_class, args.cvaemethod).to(device)
cvae.apply(weight_init)
optimizer = torch.optim.AdamW(cvae.parameters(), lr=1e-3, weight_decay=5e-4)
# optimizer = torch.optim.SGD(cvae.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, min_lr=1e-5)

# # # Training function
# def train(epoch):
#     cvae.train()
#     train_loss = 0
#     for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
#         data = data.to(device)
#         label = label.to(device)

#         cond = F.one_hot(label, num_classes=n_class).float()
        
#         optimizer.zero_grad()
#         recon_batch, mu, log_var = cvae(data, cond)
        
#         # print(f"data.shape: {data.shape}, recon_batch.shape: {recon_batch.shape}, mu.shape: {mu.shape}, log_var.shape: {log_var.shape}")

#         beta = 1
#         # Calculate losses
#         BCE = F.mse_loss(recon_batch, data.view(-1, n_feature), reduction='sum')
#         KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
#         # print(f"BCE: {BCE.item()}, KLD: {KLD.sum().item()}")
#         loss = BCE + beta * KLD.sum()
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
    
#     avg_loss = train_loss / len(train_loader.dataset)
#     print(f'Epoch {epoch}, Average loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
#     scheduler.step(avg_loss)
#     return avg_loss

# # Training loop
# losses = []
# start_time = time.time()
# for epoch in range(1, args.epochs + 1):
#     loss = train(epoch)
#     losses.append(loss)

# print(f"Training time: {time.time() - start_time:.2f} seconds")

# # Save models
# torch.save(cvae.encoder.state_dict(), f'cvae_encoder_{args.cvaemethod}.pth')
# torch.save(cvae.decoder.state_dict(), f'cvae_decoder_{args.cvaemethod}.pth')

def load_cvae():
    encoder = Encoder(args.latent, n_class, args.cvaemethod)
    decoder = Decoder(args.latent, n_class)
    
    encoder.load_state_dict(torch.load(f'cvae_encoder_{args.cvaemethod}.pth'))
    decoder.load_state_dict(torch.load(f'cvae_decoder_{args.cvaemethod}.pth'))
    
    cvae = CVAE(args.latent, n_class, args.cvaemethod)
    cvae.encoder = encoder.to(device)
    cvae.decoder = decoder.to(device)
    cvae.to(device)

    return cvae

cvae = load_cvae()

# Plot training loss
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.savefig('cvae_training_loss.png')

# Sample generation and reconstruction (example)
def plot_images():
    cvae.eval()
    with torch.no_grad():
        # Sample real images
        real_images, real_labels = next(iter(train_loader))
        real_images = real_images[:10].to(device)
        real_labels = real_labels[:10].to(device)
        
        # Get conditions
        cond = F.one_hot(real_labels, num_classes=n_class).float()
        
        # Reconstruct images
        mu, log_var = cvae.encoder(real_images, cond)
        z = cvae.reparameterize(mu, log_var)
        recon_images = cvae.decoder(z, cond)
        
        # Generate a new z from normal distribution
        z_random = torch.randn_like(z)
        # z_random = cvae.reparameterize(torch.tensor(mu_avg).to(device), torch.tensor(log_var_avg).to(device))
        # make 10 duplicates
        gen_images = cvae.decoder(z_random, cond)
        
        # Plot images
        fig, axs = plt.subplots(3, 10, figsize=(20, 6))
        for i in range(10):
            axs[0, i].imshow(real_images[i].cpu().permute(1, 2, 0))
            axs[1, i].imshow(recon_images[i].cpu().view(img_shape).permute(1, 2, 0))
            axs[2, i].imshow(gen_images[i].cpu().view(img_shape).permute(1, 2, 0))
            axs[0, i].axis('off')
            axs[1, i].axis('off')
            axs[2, i].axis('off')
        plt.savefig('cvae_images_4.png')

plot_images()