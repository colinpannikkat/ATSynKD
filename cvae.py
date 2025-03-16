from models import CVAE, ResCVAE
import torch
import torch.nn.functional as F
from utils import Datasets
from losses import elbo_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# cvae = CVAE(input_dim=3*32*32, latent_dim=20, output_dim=3*32*32, num_classes=10)  # Example for MNIST
cvae = ResCVAE(num_classes=100)
cvae = cvae.to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

num_epochs = 200
num_classes = 10  # Adjust as needed
latent_dim = 20

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# Get data
train_data, test_data = Datasets().load_cifar100(batch_size=256, n=50)

# # print("Training with soft labels:")
for epoch in tqdm(range(num_epochs)):
    cvae.train()  # Set the model to training mode
    for x_batch, y_batch in train_data:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # y_hard = F.one_hot(torch.tensor(y_batch), num_classes=num_classes).float().to(device)  # Get soft labels from CNN
        y_hard = y_batch

        x_recon, mu, logvar = cvae(x_batch, y_hard)
        loss, recon_loss, kl_loss = elbo_loss(x_batch, x_recon, mu, logvar)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN detected! Recon: {recon_loss.item()}, KL: {kl_loss.item()}")
            break  # Stop training if NaN appears

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step(loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# save the model
torch.save(cvae.state_dict(), 'cvae_cifar100.pth')


# Generate samples for all digits 0-9 with hard labels
cvae.load_state_dict(torch.load('cvae_cifar100.pth'))
fig, axes = plt.subplots(1, 10, figsize=(15, 4))


cvae.eval()  # Set the model to evaluation mode
x, y = next(iter(test_data))
x = x.to(device)
y = y.to(device)
print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")
with torch.no_grad():
    z = torch.rand_like(cvae.encoder(x, y)[0])
    x_recon = cvae.decoder(z, y)
    x_recon = x_recon.cpu().detach().numpy()
    x_recon = np.transpose(x_recon,
        (0, 2, 3, 1))  # Transpose to (batch_size, height, width, channels) for plotting
    for i in range(10):
        axes[i].imshow(x_recon[i])
        axes[i].axis('off')
        axes[i].set_title(f"Class {i}")
plt.tight_layout()
plt.savefig("./cvae_cifar100.png")