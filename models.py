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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = None
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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
        # x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)

        x = self.avgpool(g2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, [F(g) for g in (g0, g1, g2)]

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
            try:
                model_resnet32.load_state_dict(weights)
            except Exception as e:
                model_resnet32.load_state_dict(weights['model_state_dict'])

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
            try:
                model_resnet20.load_state_dict(weights)
            except Exception as e:
                model_resnet20.load_state_dict(weights['model_state_dict'])

    return model_resnet20

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1   = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2   = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1   = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2   = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))
    
class ResEncoder(nn.Module):
    '''
    Encoder block

    Built for a 3x32x32 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 32
    For images sized 32 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    '''
    def __init__(self, channels, ch=64, latent_channels=512, num_classes=10):
        super(ResEncoder, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.conv_in = nn.Conv2d(channels + num_classes, ch, 7, 1, 3)

        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.res_down_block3 = ResDown(4 * ch, 8 * ch)

        self.conv_mu = nn.Conv2d(8 * ch, latent_channels, 4, 1)
        self.conv_log_var = nn.Conv2d(8 * ch, latent_channels, 4, 1)

        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        label_emb = self.label_embedding(y)
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        
        x = torch.cat([x, label_emb], dim=1)
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)
        x = self.res_down_block2(x)
        x = self.res_down_block3(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training:
            x = self.sample(mu, log_var)
        else:
            x = mu
        return x, mu, log_var

class ResDecoder(nn.Module):
    def __init__(self, channels, ch=64, latent_channels=512, num_classes=10):
        super(ResDecoder, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # ConvTranspose receives latent_channels + num_classes channels.
        self.conv_t_up = nn.ConvTranspose2d(latent_channels + num_classes, ch * 8, 4, 1)

        self.res_up_block1 = ResUp(ch * 8, ch * 4)
        self.res_up_block2 = ResUp(ch * 4, ch * 2)
        self.res_up_block3 = ResUp(ch * 2, ch)

        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, z, y):
        label_emb = self.label_embedding(y)
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)
        label_emb = label_emb.expand(-1, -1, z.size(2), z.size(3))

        x = torch.cat([z, label_emb], dim=1)
        x = self.act_fnc(self.conv_t_up(x))

        x = self.res_up_block1(x)
        x = self.res_up_block2(x)
        x = self.res_up_block3(x)

        x = torch.sigmoid(self.conv_out(x))
        return x

class ResCVAE(nn.Module):
    def __init__(self, channel_in=3, ch=64, latent_channels=512, num_classes=10):
        super(ResCVAE, self).__init__()
        self.encoder = ResEncoder(channels=channel_in, ch=ch, latent_channels=latent_channels, num_classes=num_classes)
        self.decoder = ResDecoder(channels=channel_in, ch=ch, latent_channels=latent_channels, num_classes=num_classes)

    def forward(self, x, y):
        encoding, mu, log_var = self.encoder(x, y)
        recon_img = self.decoder(encoding, y)
        return recon_img, mu, log_var
    
class Encoder(nn.Module):
    def __init__(self, num_classes, image_shape, latent_dim, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.project = nn.Sequential(
            nn.Linear(in_features=num_classes, out_features=image_shape[1] * image_shape[2] * 1),
            nn.Unflatten(1, (1, image_shape[1], image_shape[2]))
        )
        self.concat = lambda x, label: torch.cat([x, label], dim=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=image_shape[0] + 1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        shape_post_upsample = (
            image_shape[0],
            image_shape[1] // (2 ** 5),
            image_shape[2] // (2 ** 5)
        )

        self.flatten = nn.Flatten()
        self.encode = nn.Sequential(
            nn.Linear(in_features=512 * shape_post_upsample[1] * shape_post_upsample[2], out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        self.mu = nn.Linear(in_features=16, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=16, out_features=latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, label):
        projected_label = self.project(label)
        x_w_label = self.concat(x, projected_label)
        x_w_label = self.conv1(x_w_label)
        x_encoded = self.upsample(x_w_label)
        x_encoded_flat = self.flatten(x_encoded)
        x_dense = self.encode(x_encoded_flat)
        z_mu = self.mu(x_dense)
        z_log_var = self.log_var(x_dense)
        z = self.reparameterize(z_mu, z_log_var)
        z_cond = self.concat(z, label)
        return z_cond, z_mu, z_log_var
    
class Decoder(nn.Module):
    def __init__(self, input_shape, image_shape, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        shape_post_upsample = (
            image_shape[0],
            image_shape[1] // (2 ** 5),
            image_shape[2] // (2 ** 5)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=512 * shape_post_upsample[1] * shape_post_upsample[2])
        )
        self.reshape = lambda x: x.view(-1, 512, shape_post_upsample[1], shape_post_upsample[2])
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.decode = nn.Sequential(
            nn.Linear(in_features=3 * image_shape[1] * image_shape[2], out_features=image_shape[0] * image_shape[1] * image_shape[2]),
            nn.Sigmoid()
        )
    
    def forward(self, z_cond):
        z = self.dense(z_cond)
        z = self.reshape(z)
        z = self.downsample(z)
        z = self.flatten(z)
        x_decoded = self.decode(z)
        return x_decoded
    
class CVAE(nn.Module):
    def __init__(self, num_classes, latent_dim, image_shape, *args, **kwargs):
        super(CVAE, self).__init__(*args, **kwargs)
        
        self.encoder = Encoder(num_classes, image_shape, latent_dim)
        self.decoder = Decoder(input_shape=num_classes+latent_dim,
                                  image_shape=image_shape)
        
    def forward(self, x, label):
        z_cond, z_mu, z_log_var = self.encoder(x, label)
        x_recon = self.decoder(z_cond)
        return x_recon, z_mu, z_log_var