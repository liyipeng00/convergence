'''CNN for CIFAR-10/100
https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py
'''
import torch
from torch import nn

class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*5*5, 384)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(384, 192) 
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = CNNCifar()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32

