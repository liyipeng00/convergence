import torch
import torch.nn as nn

from .wrnnsgn_cifar import wrnnsgn8, wrnnsgn8k2, wrnnsgn8k4, wrnnsgn8k8, wrnnsgn8k10, wrnnsgn14k2, wrnnsgn14k4, wrnnsgn14k8, wrnnsgn20, wrnnsgn20k2, wrnnsgn20k4, wrnnsgn20k8, wrnnsgn10, wrnnsgn10k2, wrnnsgn10k4, wrnnsgn10k8
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

def wrnnsgn8_split(num_classes=10, split=2):
    model = wrnnsgn8(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn8k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrnnsgn8k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn8k4_split(num_classes=10, split=2):
    model = wrnnsgn8k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn8k8_split(num_classes=10, split=2):
    model = wrnnsgn8k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn8k10_split(num_classes=10, split=2):
    model = wrnnsgn8k10(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrnnsgn14k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrnnsgn14k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn14k4_split(num_classes=10, split=2):
    model = wrnnsgn14k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn14k8_split(num_classes=10, split=2):
    model = wrnnsgn14k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrnnsgn20_split(num_classes=10, split=2):
    model = wrnnsgn20(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn20k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrnnsgn20k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn20k4_split(num_classes=10, split=2):
    model = wrnnsgn20k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn20k8_split(num_classes=10, split=2):
    model = wrnnsgn20k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrnnsgn10_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrnnsgn10(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn10k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrnnsgn10k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn10k4_split(num_classes=10, split=2):
    model = wrnnsgn10k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrnnsgn10k8_split(num_classes=10, split=2):
    model = wrnnsgn10k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


if __name__ == '__main__':
    from model_utils import BuildClient, BuildServer
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model = wrnnsgn20k4()
    model_client, model_server = wrnnsgn20k2_split(num_classes=10, split=2)
    print(model_client)
    print(model_server)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32