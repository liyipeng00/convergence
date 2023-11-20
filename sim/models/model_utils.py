import torch
import torch.nn as nn

class BuildClient(nn.Module):
    def __init__(self, model):
        super(BuildClient, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        self.activation = out
        return out
    
    def get_activation(self):
        return self.activation


class BuildClientAux(nn.Module):
    def __init__(self, frontnet, auxnet):
        super(BuildClientAux, self).__init__()
        self.frontnet = frontnet
        self.auxnet = auxnet
        
    def forward(self, x, part=0):
        # front net
        if part == 0:
            out = self.frontnet(x)
            self.activation = out
            return out
        # aux net
        elif part == 1:
            out = self.auxnet(x)
            return out
        else:
            raise ValueError 
    
    def get_activation(self):
        return self.activation


class BuildServer(nn.Module):
    def __init__(self, model):
        super(BuildServer, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)
        

def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params