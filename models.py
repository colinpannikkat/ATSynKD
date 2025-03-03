from torchvision.models import resnet152, resnet34, ResNet152_Weights, ResNet34_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return [F(g) for g in (g0, g1, g2, g3)], x
    
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
        x = self.fc(x)
        
        return [F(g) for g in (g0, g1, g2, g3)], x
    
def load_resnet152(dataset: str, weights = None) -> ResNet152AT:
    model_resnet152 = None
    if dataset == "imagenet":
        base_resnet152 = resnet152(weights=ResNet152_Weights.DEFAULT)
        model_resnet152 = ResNet152AT(Bottleneck, [3, 8, 36, 3])
        model_resnet152.load_state_dict(base_resnet152.state_dict())
    if dataset == "cifar10":
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
    if dataset == "cifar10":
        model_resnet34 = ResNet34AT(BasicBlock, [3, 4, 6, 3])
        if weights is not None:
            model_resnet34.load_state_dict(weights)

    return model_resnet34