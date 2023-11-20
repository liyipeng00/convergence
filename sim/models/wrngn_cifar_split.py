import torch
import torch.nn as nn

from .wrngn_cifar import wrngn8k2, wrngn8k4, wrngn8k8, wrngn8k10, wrngn14k2, wrngn14k4, wrngn14k8, wrngn20k2, wrngn20k4, wrngn20k8, wrngn10, wrngn10k2, wrngn10k4, wrngn10k8
from .model_utils import BuildClient, BuildServer


def splitmodel(model, split):
    r'''
    Args:
        split: cut layer. split = 1,2
    Note that when `split`=2, the client-side model has two parts, conv1 and layer1 (1 conv1+3 blocks for ResNet20, 1 conv1+1 blocks for ResNet8)
    '''
    if split == 1:
        net_client = nn.Sequential(model.conv1, model.bn1, model.relu)
        if hasattr(model, 'layer4'):
            net_server = nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool, model.flatten,  model.fc)  
        else:
            net_server = nn.Sequential(model.layer1, model.layer2, model.layer3, model.avgpool, model.flatten,  model.fc)  
    elif split == 2:
        net_client = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        if hasattr(model, 'layer4'):
            net_server = nn.Sequential(model.layer2, model.layer3, model.layer4, model.avgpool, model.flatten, model.fc)
        else:
            net_server = nn.Sequential(model.layer2, model.layer3, model.avgpool, model.flatten, model.fc)
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)
    return model_client, model_server


def wrngn8k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrngn8k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn8k4_split(num_classes=10, split=2):
    model = wrngn8k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn8k8_split(num_classes=10, split=2):
    model = wrngn8k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn8k10_split(num_classes=10, split=2):
    model = wrngn8k10(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrngn14k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrngn14k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn14k4_split(num_classes=10, split=2):
    model = wrngn14k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn14k8_split(num_classes=10, split=2):
    model = wrngn14k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrngn20k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrngn20k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn20k4_split(num_classes=10, split=2):
    model = wrngn20k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn20k8_split(num_classes=10, split=2):
    model = wrngn20k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrngn10_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrngn10(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn10k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrngn10k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn10k4_split(num_classes=10, split=2):
    model = wrngn10k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrngn10k8_split(num_classes=10, split=2):
    model = wrngn10k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


if __name__ == '__main__':
    from resnetgn_cifar import resnetgn20
    from model_utils import BuildClient, BuildServer
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model = wrngn20k4()
    model_client, model_server = wrngn20k2_split(num_classes=10, split=2)
    print(model_client)
    print(model_server)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32