import torch
import torch.nn as nn

from .wrn_cifar import wrn8, wrn8k2, wrn8k4, wrn8k8, wrn8k10, wrn14k2, wrn14k4, wrn14k8, wrn20, wrn20k2, wrn20k4, wrn20k6, wrn20k8, wrn10, wrn10k2, wrn10k4, wrn10k8, wrn32, wrn44, wrn56, wrn110
from .model_utils import BuildClient, BuildServer


def splitmodel(model, split):
    r'''
    Args:
        split: cut layer. split = 1,2
    Note that when `split`=2, the client-side model has two parts, conv1 and layer1 (1 conv1+3 blocks for ResNet20, 1 conv1+1 blocks for ResNet8)
    '''
    if split == 1:
        net_client = nn.Sequential(model.conv1, model.relu)
        if hasattr(model, 'layer4'):
            net_server = nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool, model.flatten,  model.fc)  
        else:
            net_server = nn.Sequential(model.layer1, model.layer2, model.layer3, model.avgpool, model.flatten,  model.fc)  
    elif split == 2:
        net_client = nn.Sequential(model.conv1, model.relu, model.layer1)
        if hasattr(model, 'layer4'):
            net_server = nn.Sequential(model.layer2, model.layer3, model.layer4, model.avgpool, model.flatten, model.fc)
        else:
            net_server = nn.Sequential(model.layer2, model.layer3, model.avgpool, model.flatten, model.fc)
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)
    return model_client, model_server

def wrn8_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn8(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn8k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn8k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn8k4_split(num_classes=10, split=2):
    model = wrn8k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn8k8_split(num_classes=10, split=2):
    model = wrn8k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn8k10_split(num_classes=10, split=2):
    model = wrn8k10(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrn14k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn14k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn14k4_split(num_classes=10, split=2):
    model = wrn14k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn14k8_split(num_classes=10, split=2):
    model = wrn14k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrn20_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn20(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn20k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn20k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn20k4_split(num_classes=10, split=2):
    model = wrn20k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn20k6_split(num_classes=10, split=2):
    model = wrn20k6(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn20k8_split(num_classes=10, split=2):
    model = wrn20k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


def wrn32_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn32(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn44_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn44(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn56_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn56(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn110_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn110(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn10_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn10(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn10k2_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = wrn10k2(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn10k4_split(num_classes=10, split=2):
    model = wrn10k4(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def wrn10k8_split(num_classes=10, split=2):
    model = wrn10k8(num_classes=num_classes)
    return splitmodel(model=model, split=split)


if __name__ == '__main__':
    from model_utils import BuildClient, BuildServer
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model = wrn20k4()
    model_client, model_server = wrn20k2_split(num_classes=10, split=2)
    print(model_client)
    print(model_server)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32