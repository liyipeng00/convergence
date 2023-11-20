import torch
import torch.nn as nn

from .resnetii_cifar import resnetii10, resnetii18
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

def resnetii10_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = resnetii10(num_classes=num_classes)
    return splitmodel(model=model, split=split)

def resnetii18_split(num_classes=10, split=2):
    r'''split: cut layer. split = 1,2*'''
    model = resnetii18(num_classes=num_classes)
    return splitmodel(model=model, split=split)


if __name__ == '__main__':
    from model_utils import BuildClient, BuildServer
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model = resnetii10()
    model_client, model_server = resnetii10_split(num_classes=10, split=2)
    print(model_client)
    print(model_server)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32