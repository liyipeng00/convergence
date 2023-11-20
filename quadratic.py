import torch
import torch.nn as nn
from torch.optim import SGD

import numpy as np
import copy
import os

from sim.utils.utils import average_weights, setup_seed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-R', default=200, type=int, help='Number of total training rounds')
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=100, type=int, help='Number of clients participate')
parser.add_argument('--F1', default=1, type=float, nargs='*', help='F1')
parser.add_argument('--F2', default=1, type=float, nargs='*', help='F2')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--momentum', default=0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--seed', default=1234, type=int, help='seed')
parser.add_argument('--device', default=0, type=int, help='Device')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

def record_setup(args, alg):
    '''Setup format: 
    quadratic_PFL_F1_0.50,1.00_F2_0.50,-1.00_M2_K10_R500_sgd0.06,0.0,0.0_seed1234
    '''
    setup = 'quadratic_{}_F1_{:.2f},{:.2f}_F2_{:.2f},{:.2f}_M{}_K{}_R{}_sgd{},{},{}_seed{}' \
                        .format(alg, args.F1[0], args.F1[1], args.F2[0], args.F2[1], args.M, args.K, args.R, \
                                args.lr, args.momentum, args.weight_decay, args.seed)
    return setup

def record_exp_result(filename, result, round):
    savepath = './save/'
    filepath = '{}/{}.csv'.format(savepath, filename)
    if round == 0:
        if (os.path.exists(filepath)):
            os.remove(filepath)
        with open (filepath, 'a+') as f:
            f.write('{},{}\n'.format('round', 'distance'))
            f.write('{},{:.4f}\n'.format(round, result))
    else:
        with open (filepath, 'a+') as f:
            f.write('{},{:.4f}\n'.format(round, result))

class QuadraticFunc(nn.Module):
    def __init__(self, in_dim=1):
        super(QuadraticFunc, self).__init__()
        #self.x = torch.nn.Parameter(torch.randn((in_dim))[0])
        self.x = torch.nn.Parameter(torch.tensor(10, dtype=float))

    def forward(self, data=[1,1,1]):
        out = data[0] * self.x**2 + data[1] * self.x + data[2]
        return out

def local_update(args, data, model):
    global device
    
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model.train()
    data.to(device)
    num_steps = args.K
    step_count = 0
    while(True):
        local_objective = model(data)
        optimizer.zero_grad()
        local_objective.backward()
        optimizer.step()
        step_count +=1
        if (step_count >= num_steps):
                break
    return model

def parallel_train(args, data, optim):
    global device

    global_model = QuadraticFunc()
    global_model.to(device)
    global_model.train()

    datasetsize_client = [1 for _ in range(len(data))]
    
    with torch.no_grad():
        distance = float(torch.norm(global_model.x.data - optim).cpu().numpy())
        record_exp_result(record_setup(args, 'PFL'), distance, 0)

    for r in range(args.R):
        weight_list = []
        active_clients = range(2)
        active_datasetsize = []
        for u in active_clients:
            local_model = local_update(args, data[u], model=copy.deepcopy(global_model))
            weight_list.append(local_model.state_dict())
            active_datasetsize.append(datasetsize_client[u])

        average_weight = average_weights(weight_list, active_datasetsize) # Note: use active_datasetsize
        global_model.load_state_dict(average_weight)
        
        with torch.no_grad():
            distance = float(torch.norm(global_model.x.data - optim).cpu().numpy())
            record_exp_result(record_setup(args, 'PFL'), distance, r+1)

def sequential_train(args, data, optim):
    global device

    global_model = QuadraticFunc()
    global_model.to(device)
    global_model.train()

    with torch.no_grad():
        distance = float(torch.norm(global_model.x.data - optim).cpu().numpy())
        record_exp_result(record_setup(args, 'SFL'), distance, 0)

    for r in range(args.R):
        local_model = copy.deepcopy(global_model)
        active_clients = np.random.choice(range(2), 2, replace=False)
        for u in active_clients:
            local_model = local_update(args, data[u], model=local_model)
        global_model.load_state_dict(local_model.state_dict())

        with torch.no_grad():
            distance = float(torch.norm(global_model.x.data - optim).cpu().numpy())
            record_exp_result(record_setup(args, 'SFL'), distance, r+1)


def train(args, data):
    setup_seed(args.seed)

    optim = - 0.5 * (data[0][1] + data[1][1]) / (data[0][0] + data[1][0]) 
    parallel_train(args, data, optim)
    sequential_train(args, data, optim)

# python quadratic.py -R 500 -K 2 -M 2 -P 2 --F1 0.5 1 --F2 0.5 -1 --lr 0 --momentum 0 --weight-decay 0  --seed 0 
# 

def get_result():
    global args
    
    #data1 = torch.tensor([1/2, 1, 0], dtype=float)
    #data2 = torch.tensor([1/2, -1, 0], dtype=float)
    data1 = torch.tensor([args.F1[0], args.F1[1], 0], dtype=float)
    data2 = torch.tensor([args.F2[0], args.F2[1], 0], dtype=float)
    data = [data1, data2]

    #lrs = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6]
    lrs = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
    #seeds = [123, 1234, 12345]
    seeds = [1, 12, 123, 1234, 12345]

    for lr in lrs:
        args.lr = lr
        for seed in seeds:
            args.seed = seed
            train(args, data)

def main():
    get_result()

if __name__ == '__main__':
    main()