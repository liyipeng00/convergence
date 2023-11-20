'''CNN for MNIST, FashionMNIST
https://github.com/IBM/fl-arbitrary-participation/blob/main/model/cnn_mnist.py
2023-08-26
'''
import torch
from torch import nn

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(128, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = CNNMnist()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32