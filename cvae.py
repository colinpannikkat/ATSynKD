from models import CVAE, ResCVAE
import torch
import torch.nn.functional as F
from utils import Datasets
from losses import elbo_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# cvae = CVAE(input_dim=3*32*32, latent_dim=20, output_dim=3*32*32, num_classes=10)  # Example for MNIST
cvae = ResCVAE(num_classes=100)
cvae = cvae.to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

num_epochs = 200
num_classes = 10  # Adjust as needed
latent_dim = 20

# Training loop with soft labels
optimizer_soft = torch.optim.AdamW(cvae.parameters(), lr=1e-3)

# Get data
train_data, test_data = Datasets().load_cifar100(batch_size=256)

print("Training with soft labels:")
for epoch in tqdm(range(num_epochs)):
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

        optimizer_soft.zero_grad()
        loss.backward()
        optimizer_soft.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

# Generate samples for all digits 0-9 with hard labels
fig, axes = plt.subplots(1, 10, figsize=(15, 4))

cvae.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for i in range(10):
        z = torch.randn(1, 20).to(device)  # Sample from standard normal distribution
        y_hard = torch.zeros(1, 10).to(device)
        y_hard[0, i] = 1.0  # Generate a sample for class i
        generated_sample = cvae.decoder(z, y_hard)
        generated_sample = generated_sample.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()  # Reshape to RGB image and move to CPU
        
        axes[i].imshow(generated_sample)
        axes[i].set_title(f"Hard Class {i}")
        axes[i].axis('off')

plt.savefig("./")