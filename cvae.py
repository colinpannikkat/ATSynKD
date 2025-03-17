from models import CVAE
from losses import ELBOLoss
import torch
import torch.nn.functional as F
from utils import Datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

num_epochs = 600
num_classes = 100  # Adjust as needed
latent_dim = 3
batch=64
beta=1
image_shape = (3, 32, 32)

cvae = CVAE(
    num_classes=num_classes,
    latent_dim=latent_dim,
    image_shape=image_shape
)
cvae = cvae.to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
criterion = ELBOLoss(beta=beta, reduction='sum')
scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, min_lr=1e-12)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)

# Get data
train_data, test_data, _ = Datasets().load_cifar100(batch_size=batch, n=50, wait_transform=True)

best_model = (None, float('inf'), None)  # (model_state_dict, best_loss, epoch)

for epoch in tqdm(range(num_epochs)):
    cvae.train()  # Set the model to training mode
    for x_batch, y_batch in train_data:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_hard = F.one_hot(y_batch, num_classes=num_classes).float().to(device)

        x_recon, mu, logvar = cvae(x_batch, y_hard)
        x_recon = x_recon.view(-1, image_shape[0], image_shape[1], image_shape[2])
        loss, recon_loss, kl_loss = criterion(x_batch, x_recon, mu, logvar)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN detected! Recon: {recon_loss.item()}, KL: {kl_loss.item()}")
            break  # Stop training if NaN appears

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step(loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Recon: {recon_loss.item():.10f}, KL: {kl_loss.item():.10f}, LR: {optimizer.param_groups[0]['lr']:.10f}")

    # Save the best model
    if loss.item() < best_model[1]:
        best_model = (cvae.state_dict(), loss.item(), epoch + 1)

# Save the best model to a file
if best_model[0] is not None:
    torch.save(best_model[0], 'cvae_cifar10.pth')
    print(f"Best model saved with loss {best_model[1]:.6f} at epoch {best_model[2]}")

# Generate samples for all digits 0-9 with hard labels
cvae.load_state_dict(torch.load('cvae_cifar10.pth'))
fig, axes = plt.subplots(1, 10, figsize=(15, 4))

cvae.eval()  # Set the model to evaluation mode
x, y = next(iter(test_data))
y_hot = F.one_hot(torch.tensor(y), num_classes=num_classes).float().to(device)
x = x.to(device)
y_hot = y_hot.to(device)
print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")
with torch.no_grad():
    z = torch.randn((batch, latent_dim)).to(device)
    z = torch.cat([z, y_hot], dim=1)
    x_recon = cvae.decoder(z)
    x_recon = x_recon.cpu().detach()
    x_recon = x_recon.view(-1, image_shape[0], image_shape[1], image_shape[2]).numpy()
    x_recon = np.transpose(x_recon,
        (0, 2, 3, 1))  # Transpose to (batch_size, height, width, channels) for plotting
    for i in range(10):
        axes[i].imshow(x_recon[i])
        axes[i].axis('off')
        axes[i].set_title(f"Class {y[i]}")
plt.tight_layout()
plt.savefig("./cvae_cifar10.png")