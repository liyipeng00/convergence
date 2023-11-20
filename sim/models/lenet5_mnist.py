'''LeNet5 for MNIST, FashionMNIST
modified from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
2022-08-04
'''
import torch
import torch.nn as nn
# Note: comment the relative import if you want to run in this file (__name__ == '__main__').

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        y = self.features(x)
        y = self.flatten(y)
        y = self.classifier(y)
        return y


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