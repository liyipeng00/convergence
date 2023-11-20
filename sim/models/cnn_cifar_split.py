'''CNN for CIFAR-10/100
https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py
'''
import torch
from torch import nn
from .cnn_cifar import CNNCifar
from .model_utils import BuildClient, BuildServer    

def splitmodel(model, split):
    if split == 1:
        net_client = nn.Sequential(model.conv1, model.relu1, model.pool1)
        net_server = nn.Sequential(model.conv2, model.relu2, model.pool2, model.flatten, model.fc1, model.relu3, model.fc2, model.relu4, model.fc3)
    elif split == 2:
        net_client = nn.Sequential(model.conv1, model.relu1, model.pool1, model.conv2, model.relu2, model.pool2)
        net_server = nn.Sequential(model.flatten, model.fc1, model.relu3, model.fc2, model.relu4, model.fc3)
        
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)

    return model_client, model_server


def cifarcnn_split(num_classes=10, split=2):
    r'''
    Args:
        split: cut layer. split = 1,2*
    '''
    model = CNNCifar(num_classes)
    return splitmodel(model=model, split=split)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = CNNCifar()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32

