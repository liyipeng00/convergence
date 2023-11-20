'''
Split ResNet(CIFAR10/100) to client model and server model.
The split method are from [1].
References:
[1] https://github.com/ZHUANGHP/FDG/blob/master/DG_models.py
[2022 05 02]
'''
import torch
import torch.nn as nn

from .resnetgn_cifar import resnetgn20, resnetgn14, resnetgn8
from .model_utils import BuildClient, BuildServer


def splitmodel(model, split):
    r'''
    Args:
        split: cut layer. split = 1,2
    Note that when `split`=2, the client-side model has two parts, conv1 and layer1 (1 conv1+3 blocks for ResNet20, 1 conv1+1 blocks for ResNet8)
    '''
    if split == 1:
        net_client = nn.Sequential(model.conv1, model.bn1, model.relu)
        net_server = nn.Sequential(model.layer1, model.layer2, model.layer3, model.avgpool, model.flatten,  model.fc)  
    elif split == 2:
        net_client = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        net_server = nn.Sequential(model.layer2, model.layer3, model.avgpool, model.flatten, model.fc)
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)
    return model_client, model_server


# def resnetgn4_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn4(num_classes=num_classes)
#     if split == 1:
#         net_client = nn.Sequential(model.conv1, model.bn1, model.relu)
#         net_server = nn.Sequential(model.layer1, model.avgpool, model.flatten,  model.fc)
#     model_client = BuildClient(net_client)
#     model_server = BuildServer(net_server)
#     return model_client, model_server

# def resnetgn6_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn6(num_classes=num_classes)
#     if split == 1:
#         net_client = nn.Sequential(model.conv1, model.bn1, model.relu)
#         net_server = nn.Sequential(model.layer1, model.layer2, model.avgpool, model.flatten,  model.fc)
#     model_client = BuildClient(net_client)
#     model_server = BuildServer(net_server)
#     return model_client, model_server

def resnetgn8_split(num_classes=10, split=2):
    model = resnetgn8(num_classes=num_classes)
    return splitmodel(model, split)
    
def resnetgn14_split(num_classes=10, split=2):
    model = resnetgn14(num_classes=num_classes)
    return splitmodel(model, split)

def resnetgn20_split(num_classes=10, split=2):
    r'''
    Args:
        split: cut layer. split = 1,2*
    '''
    model = resnetgn20(num_classes=num_classes)
    return splitmodel(model=model, split=split)

# def resnetgn26_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn26(num_classes=num_classes)
#     return splitmodel1(model, split)

# def resnetgn32_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn32(num_classes=num_classes)
#     return splitmodel1(model, split)

# def resnetgn44_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn44(num_classes=num_classes)
#     return splitmodel1(model, split)

# def resnetgn56_split(split, dtype, num_classes, **kwargs):
#     model = resnetgn56(num_classes=num_classes)
#     return splitmodel1(model, split)

if __name__ == '__main__':
    from resnetgn_cifar import resnetgn20
    from model_utils import BuildClient, BuildServer
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model = resnetgn20()
    model_client, model_server = resnetgn20_split(num_classes=10, split=4)
    print(model_client)
    print(model_server)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32