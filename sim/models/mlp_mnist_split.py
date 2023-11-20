import torch
from torch import nn
from .model_utils import BuildClient, BuildServer
from .mlp_mnist import MLP

def mlp_split(num_classes=10, split=1):
    '''
    To be consistent with other models, we keep `num_classes`.
    Args:
        split: cut layer. split = 1
    '''
    assert num_classes==10
    model = MLP()
    if split == 1:
        net_client = nn.Sequential(model.flatten, model.layer_input, model.dropout, model.relu) 
        net_server = nn.Sequential(model.layer_hidden)      
    model_client = BuildClient(net_client) # front net
    model_server = BuildServer(net_server) # end net
    return model_client, model_server


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = MLP()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32