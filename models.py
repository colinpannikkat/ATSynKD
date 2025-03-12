from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
from torchvision.models import resnet34, ResNet34_Weights

class ResNetAT(ResNet):
    """Attention maps of ResNet for a teacher model.
    
    Overloaded ResNet model to return attention maps and also handle 3 residual
    layers instead of the normal four (per CIFAR-10 implementation).
    """
    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = None
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x, F: Callable = lambda x: x.pow(2).sum(dim=1)):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)

        x = self.avgpool(g2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return [F(g) for g in (g0, g1, g2)], x

def load_resnet32(dataset: str, weights = None) -> ResNetAT:
    model_resnet32 = None
    if dataset == "cifar10":
        model_resnet32 = ResNetAT(BasicBlock, [5, 5, 5, 0], num_classes=10)
    elif dataset == "cifar100":
        # base_resnet34 = resnet34(weights = ResNet34_Weights)
        model_resnet32 = ResNetAT(BasicBlock, [5, 5, 5, 0], num_classes=100)
        # base_dict = base_resnet34.state_dict()
        # model_dict = model_resnet32.state_dict()
        # model_dict.update({k: v for k, v in base_dict.items() if k in model_dict and v.shape == model_dict[k].shape})
        # model_resnet32.load_state_dict(model_dict)
    elif dataset == "tiny-imagenet":
        model_resnet32 = ResNetAT(BasicBlock, [5, 5, 5, 0], num_classes=200)
    else:
        raise(Exception(f"Please add clause for {dataset}."))
    
    if weights is not None:
            model_resnet32.load_state_dict(weights)

    return model_resnet32

def load_resnet20(dataset: str, weights = None) -> ResNetAT:
    model_resnet20 = None
    if dataset == "cifar10":
        model_resnet20 = ResNetAT(BasicBlock, [3, 3, 3, 0], num_classes=10)
    elif dataset == "cifar100":
        model_resnet20 = ResNetAT(BasicBlock, [3, 3, 3, 0], num_classes=100)
    elif dataset == "tiny-imagenet":
        model_resnet20 = ResNetAT(BasicBlock, [3, 3, 3, 0], num_classes=200)
    else:
        raise(Exception(f"Please add clause for {dataset}."))
    
    if weights is not None:
            model_resnet20.load_state_dict(weights)

    return model_resnet20
    
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