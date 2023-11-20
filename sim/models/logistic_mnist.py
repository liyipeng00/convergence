'''Logistic for MNIST [1].
Replace the `x = self.flatten_data(x)` in [2] with `nn.Flatten()` to conincide with other models.
References:
[1] https://github.com/lx10077/fedavgpy/blob/master/src/models/model.py
[2] https://github.com/lx10077/fedavgpy/blob/master/src/models/worker.py
2023-06-19
'''
import torch
from torch import nn

class Logistic(nn.Module):
    def __init__(self, dim_in=28*28, dim_out=10):
        super(Logistic, self).__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        logit = self.layer(x)
        return logit

if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = Logistic()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32