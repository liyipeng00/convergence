'''Multi-layer perceptron for MNIST
This code is mainly from [1]. We make some modifications according to [3], 
i.e., remove `self.softmax = nn.Softmax(dim=1)`).
As a result, we use `CrossEntropy` instead of `NLLLoss` as the loss function.
Ref [3] is another implementation of MLP (without "dropout").
References:
[1] https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch/issues/30
[3] https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
2023-02-22
'''
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_in=28*28, dim_hidden=64, dim_out=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = MLP()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32