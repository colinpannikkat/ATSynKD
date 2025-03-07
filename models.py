from torchvision.models import resnet152, resnet34, ResNet152_Weights, ResNet34_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.functional as F
from torch import nn
from typing import Callable

class ResNet152AT(ResNet):
    """Attention maps of ResNet-152.
    
    Overloaded ResNet model to return attention maps.
    """
    def __init__(self, block, layers, grayscale = False, num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

        # if grayscale:
        #     in_dim = 1
        # else:
        #     in_dim = 3
            
        # self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x, F: Callable = lambda x: x.pow(2).sum(dim=1)):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        x = self.avgpool(g3)
        embed = torch.flatten(x, 1)
        x = self.fc(embed)
        
        return [F(g) for g in (g0, g1, g2, g3)], embed, x
    
class ResNet34AT(ResNet):
    """Attention maps of ResNet-34.
    
    Overloaded ResNet model to return attention maps.
    """
    def __init__(self, block, layers, grayscale = False, num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

        # if grayscale:
        #     in_dim = 1
        # else:
        #     in_dim = 3

        # self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        # These are used to map ResNet34 output before FC to the teacher models dimension
        self.embed_conv = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x, F: Callable = lambda x: x.pow(2).sum(dim=1)):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        x = self.avgpool(g3)
        x = torch.flatten(x, 1)
        embed = self.embed_conv(x)
        x = self.fc(embed)
        
        return [F(g) for g in (g0, g1, g2, g3)], embed, x
    
def load_resnet152(dataset: str, weights = None) -> ResNet152AT:
    model_resnet152 = None
    if dataset == "imagenet":
        base_resnet152 = resnet152(weights=ResNet152_Weights.DEFAULT)
        model_resnet152 = ResNet152AT(Bottleneck, [3, 8, 36, 3])
        model_resnet152.load_state_dict(base_resnet152.state_dict())

    else:
        model_resnet152 = ResNet152AT(Bottleneck, [3, 8, 36, 3])
        if weights is not None:
            model_resnet152.load_state_dict(weights)

    return model_resnet152

def load_resnet34(dataset: str, weights = None) -> ResNet34AT:
    model_resnet34 = None
    if dataset == "imagenet":
        base_resnet34 = resnet34(weights=ResNet34_Weights.DEFAULT)
        model_resnet34 = ResNet34AT(BasicBlock, [3, 4, 6, 3])
        model_resnet34.load_state_dict(base_resnet34.state_dict())

    else:
        model_resnet34 = ResNet34AT(BasicBlock, [3, 4, 6, 3])
        if weights is not None:
            model_resnet34.load_state_dict(weights)
            
    return model_resnet34
    
class ResEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim, condition_size):
        super(Encoder, self).__init__()
        pass

    def forward(self, x, y_soft):
       pass

class ResDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, condition_size):
        super(Decoder, self).__init__()
        pass

    def forward(self, z, y_soft):
        pass

class ResCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, condition_size):
        super(ResCVAE, self).__init__()
        self.encoder = ResEncoder(input_dim, latent_dim, condition_size)
        self.decoder = ResDecoder(latent_dim, output_dim, condition_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_soft):
        mu, logvar = self.encoder(x, y_soft)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, y_soft)
        return x_recon, mu, logvar
    
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + num_classes, 400)
        self.fc2_mu = nn.Linear(400, latent_dim)  # Mean
        self.fc2_logvar = nn.Linear(400, latent_dim)  # Log-variance

    def forward(self, x, y_soft):
        x = x.view(x.size(0), -1)  # Flatten input to [batch, input_dim]
        x = torch.cat([x, y_soft], dim=1)  # Concatenate hard labels
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 400)
        self.fc2 = nn.Linear(400, output_dim)

    def forward(self, z, y_hard):
        z = torch.cat([z, y_hard], dim=1)  # Concatenate hard labels
        h = F.elu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))  # Sigmoid for reconstruction
        return x_recon

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_classes):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, output_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_soft):
        mu, logvar = self.encoder(x, y_soft)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, y_soft)
        return x_recon, mu, logvar