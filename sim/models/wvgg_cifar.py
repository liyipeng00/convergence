'''Wide (or scaling) VGG for cifar10/cifar100
Modified from https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/models/vgg.py
Model           # Param
wvgg9k1         145,210
wvgg9k2         578,410
wvgg9k4       2,308,810
wvgg9(k8)     9,225,610
'''
from typing import Union, List, Dict, Any, cast
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, k, num_classes=10, init_weights=True, dropout=0.5):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64*k, num_classes)
        # https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, k, batch_norm = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v * k) # v -> v*k
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "WA":[8, "M", 16, "M", 32, 32, "M", 64, 64, "M", 64, 64, "M"],
    "WD":[8, 8, "M", 16, 16, "M", 32, 32, 32, "M", 64, 64, 64, "M", 64, 64, 64, "M"],

    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg, k, batch_norm, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], k=k, batch_norm=batch_norm), k, **kwargs)
    return model


def wvgg9(**kwargs: Any) -> VGG:
    return _vgg("WA", 8, False, **kwargs)

def wvgg14(**kwargs: Any) -> VGG:
    return _vgg("WD", 8, False, **kwargs)


def wvgg9_bn(**kwargs: Any) -> VGG:
    return _vgg("WA", 8, True, **kwargs)

def wvgg9k1(**kwargs: Any) -> VGG:
    return _vgg("WA", 1, False, **kwargs)

def wvgg9k2(**kwargs: Any) -> VGG:
    return _vgg("WA", 2, False, **kwargs)

def wvgg9k4(**kwargs: Any) -> VGG:
    return _vgg("WA", 4, False, **kwargs)

def wvgg9k10(**kwargs: Any) -> VGG:
    return _vgg("WA", 10, False, **kwargs)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = wvgg9(num_classes=10)
    for param_tuple in model.named_parameters():
        name, param = param_tuple
        print("name ({}) = {}".format(type(name), name))
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32

'''[Depreciated, 2023 06 19]
We adopt the same hyper parameters for cifar10 and cifar100.
Other hyper parameters for cifar100:
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py
'''