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

'''Replace the Batchnorm with Groupnorm based on [4].
The group num is set to 2 according to [5,6]
[4] The Non-IID Data Quagmire of Decentralized Machine Learning
[5] https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py
[6] https://github.com/Divyansh03/FedExP/blob/main/util_models.py
[2023 06 08]'''

'''Wide ResNet modified from [7].
[7] https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/models/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
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
        #self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.bn2 = nn.GroupNorm(2, planes)

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
                     #nn.GroupNorm(2, self.expansion * planes)
                )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        #out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(ResNet, self).__init__()
        

        self.in_planes = 16*k
        self.conv1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.bn1 = nn.GroupNorm(2, 16*k)
        self.relu = nn.ReLU()
        
        # Construct resnet for any num_blocks [2023 06 08]
        # https://github.com/TsingZ0/PFL-Non-IID/blob/master/system/flcore/trainmodel/resnet.py
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # ==>
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, 128*k, num_blocks[3], stride=2)
        
        #Note self.avgpool= nn.AvgPool2d(kernel_size=6) [2022 05 02]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Construct resnet for any num_blocks [2023 06 08]
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        # ==>
        if len(num_blocks) == 4:
            self.fc = nn.Linear(128*k*block.expansion, num_classes)
        else:
            self.fc = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if hasattr(self, 'layer4'):
            out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

'''Wide ResNet8 with group norm * k = 2, 4, 8
# model                  # params
wrn8k1 (original)          75,290
wrn8k2                    298,026
wrn8k4                  1,185,866
wrn8k8                  4,731,018
'''
def wrn8(num_classes=10):
    '''2+(1+1+1)*2 = 8'''
    return ResNet(BasicBlock, [1, 1, 1], 1, num_classes)

def wrn8k2(num_classes=10):
    '''2+(1+1+1)*2 = 8'''
    return ResNet(BasicBlock, [1, 1, 1], 2, num_classes)

def wrn8k4(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], 4, num_classes)

def wrn8k8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], 8, num_classes)

def wrn8k10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], 10, num_classes)

'''Wide ResNet14 with group norm * k = 2, 4, 8
# model                  # params
wrn14k1 (original)        172,506
wrn14k2                   685,994
wrn14k4                 2,735,946
wrn14k8                10,927,754
'''
def wrn14(num_classes=10):
    '''2+(2+2+2)*2 = 14'''
    return ResNet(BasicBlock, [2, 2, 2], 1, num_classes)

def wrn14k2(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2], 2, num_classes)

def wrn14k4(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2], 4, num_classes)

def wrn14k8(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2], 8, num_classes)

'''Wide ResNet20 with group norm * k = 2, 4, 8
# model                  # params
wrn20k1 (original)        269,722
wrn20k2                 1,073,962
wrn20k4                 4,286,026
wrn20k8                17,124,490
'''
def wrn20(num_classes=10):
    '''2+(3+3+3)*2 = 20'''
    return ResNet(BasicBlock, [3, 3, 3], 1, num_classes)

def wrn20k2(num_classes=10):
    '''2+(1+1+1)*2 = 8'''
    return ResNet(BasicBlock, [3, 3, 3], 2, num_classes)

def wrn20k4(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], 4, num_classes)

def wrn20k6(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], 6, num_classes)

def wrn20k8(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], 8, num_classes)


def wrn32(num_classes=10):
    '''2+(5+5+5)*2 = 32'''
    return ResNet(BasicBlock, [5, 5, 5], 1, num_classes)

def wrn44(num_classes=10):
    '''2+(7+7+7)*2 = 44'''
    return ResNet(BasicBlock, [7, 7, 7], 1, num_classes)

def wrn56(num_classes=10):
    '''2+(9+9+9)*2 = 56'''
    return ResNet(BasicBlock, [9, 9, 9], 1, num_classes)

def wrn110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], 1, num_classes)

'''Wide ResNet10 with group norm * k = 2, 4, 8
# model                  # params
wrn10k1 (original)        269,722
wrn10k2                 1,073,962
wrn10k4                 4,286,026
wrn10k8                17,124,490
'''
def wrn10(num_classes=10):
    '''2+(1+1+1+1)*2 = 10'''
    return ResNet(BasicBlock, [1, 1, 1, 1], 1, num_classes)

def wrn10k2(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], 2, num_classes)

def wrn10k4(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], 4, num_classes)

def wrn10k8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], 8, num_classes)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = wrn110()
    for param_tuple in model.named_parameters():
        name, param = param_tuple
        print("name ({}) = {}".format(type(name), name))
    
    assert len(dict(model.named_parameters()).keys()) == len(model.state_dict().keys()), 'More BN layers are there...'

    #count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32

