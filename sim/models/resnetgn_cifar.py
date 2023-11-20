'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

'''ResNet for CIFAR10/100
The main code is from [1]. We made some modifies based on [2,3].
The hyper parameters of CIFAR10 is the same as CIFAR100 except num_classes as [2].
References:
[1] https://github.com/akamaster/pytorch_resnet_cifar10
[2] https://github.com/ZHUANGHP/FDG/blob/master/ResNet_cifar.py
[3] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
[2022 05 02]'''

'''
Replace the Batchnorm with Groupnorm based on [4].
The group num is set to 2 according to [5,6]
[4] The Non-IID Data Quagmire of Decentralized Machine Learning
[5] https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py
[6] https://github.com/Divyansh03/FedExP/blob/main/util_models.py
[2023 06 08]'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNet', 'resnetgn4', 'resnetgn6', 'resnetgn8', 'resnetgn14', 'resnetgn20', 'resnetgn26', 'resnetgn32', 'resnetgn44', 'resnetgn56', 'resnetgn110', 'resnetgn1202']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(2, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     #nn.BatchNorm2d(self.expansion * planes)
                     nn.GroupNorm(2, self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.GroupNorm(2, 16)
        self.relu = nn.ReLU()
        
        # Construct resnet for any num_blocks [2023 06 08]
        # https://github.com/TsingZ0/PFL-Non-IID/blob/master/system/flcore/trainmodel/resnet.py
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # ==>
        planes = [16, 32, 64]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        for i in range(len(num_blocks)-1):
            setattr(self, f'layer{i+2}', self._make_layer(block, planes[i+1], num_blocks[i+1], stride=2))
        
        #Note self.avgpool= nn.AvgPool2d(kernel_size=6) [2022 05 02]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Construct resnet for any num_blocks [2023 06 08]
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        # ==>
        self.fc = nn.Linear(planes[len(num_blocks)-1] * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if hasattr(self, 'layer2'):
            out = self.layer2(out)
        if hasattr(self, 'layer3'):
            out = self.layer3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

def resnetgn4(num_classes=10):
    '''2+(1)*2 = 4'''
    return ResNet(BasicBlock, [1], num_classes)

def resnetgn6(num_classes=10):
    '''2+(1+1)*2 = 6'''
    return ResNet(BasicBlock, [1, 1], num_classes)

def resnetgn8(num_classes=10):
    '''2+(1+1+1)*2 = 8'''
    return ResNet(BasicBlock, [1, 1, 1], num_classes)

def resnetgn14(num_classes=10):
    '''2+(2+2+2)*2 = 14'''
    return ResNet(BasicBlock, [2, 2, 2], num_classes)

def resnetgn20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def resnetgn26(num_classes=10):
    return ResNet(BasicBlock, [4, 4, 4], num_classes)

def resnetgn32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)

def resnetgn44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnetgn56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnetgn110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnetgn1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = resnetgn14()
    
    assert len(dict(model.named_parameters()).keys()) == len(model.state_dict().keys()), 'More BN layers are there...'

    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32

