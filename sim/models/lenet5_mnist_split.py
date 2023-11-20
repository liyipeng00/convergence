import torch
import torch.nn as nn
from .model_utils import BuildClient, BuildServer
from .lenet5_mnist import LeNet5
# Note: comment the relative import if you want to run in this file (__name__ == '__main__').

def lenet5_split(num_classes=10, split=2):
    '''
    To be consistent with other models, we keep `num_classes`.
    Args:
        split: cut layer. split = 1,2
    '''
    assert num_classes==10
    model = LeNet5()
    if split == 1:  
        net_client = nn.Sequential(model.features[0:3])
        net_server = nn.Sequential(model.features[3:6], model.flatten, model.classifier)
    elif split == 2:
        net_client = nn.Sequential(model.features[0:6])
        net_server = nn.Sequential(model.flatten, model.classifier)
    model_client = BuildClient(net_client) # front net
    model_server = BuildServer(net_server) # end net

    return model_client, model_server


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import BuildClient, BuildServer, count_parameters
    #from model_utils import count_parameters
    model = LeNet5()
    print(model)
    #model1, model2 = lenet5_split(split=2)
    #count_parameters(model)
    #print(summary(model1, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32