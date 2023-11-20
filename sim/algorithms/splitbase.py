import numpy as np
import torch
from torch.utils.data import DataLoader
from sim.utils.utils import AverageMeter, accuracy

eval_batch_size = 32

###### CLIENT ######
class SplitClient():
    def __init__(self):
        pass

    def setup_train_dataset(self, dataset):
        self.train_feddataset = dataset

    def setup_test_dataset(self, dataset):
        self.test_dataset = dataset
    
    def setup_model(self, model):
        self.global_model = model

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit

###### SERVER ######
class SplitServer():
    def __init__(self):
        pass
    
    def setup_criterion(self, criterion):
        self.criterion = criterion
    
    def setup_model(self, model):
        self.global_model = model

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit
    
    def select_clients(self, num_clients, num_clients_per_round):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
        num_clients_per_round = min(num_clients_per_round, num_clients)
        return np.random.choice(num_clients, num_clients_per_round, replace=False)

    def local_update_step(self, client_model, dataset, num_steps, client_optim_kit, device, **kwargs):
        data_loader = DataLoader(dataset, batch_size=client_optim_kit.batch_size, shuffle=True)
        client_optimizer = client_optim_kit.optim(client_model.parameters(), **client_optim_kit.settings)
        server_optimizer = self.optim_kit.optim(self.global_model.parameters(), **self.optim_kit.settings)
        
        client_model.train()
        self.global_model.train()
        step_count = 0
        while(True):
            for input, target in data_loader:
                # get the inputs; data is a list of [inputs, labels]
                input = input.to(device)
                target = target.to(device)

                # client forward training
                activation = client_model(input)
                smashed_data = activation.clone().detach().requires_grad_(True)
                    
                # server forward/backward training
                output = self.global_model(smashed_data)
                loss = self.criterion(output, target)
                server_optimizer.zero_grad()
                loss.backward()
                client_grad = smashed_data.grad.clone().detach()

                if 'clip' in kwargs.keys() and kwargs['clip'] > 0:
                    total_norm1 = torch.nn.utils.clip_grad_norm_(parameters=self.global_model.parameters(), max_norm=kwargs['clip'])
                server_optimizer.step()

                # client backward training
                client_optimizer.zero_grad()
                activation.backward(client_grad)

                if 'clip' in kwargs.keys() and kwargs['clip'] > 0:
                    total_norm2 = torch.nn.utils.clip_grad_norm_(parameters=client_model.parameters(), max_norm=kwargs['clip'])
                client_optimizer.step()

                step_count += 1
                if (step_count >= num_steps):
                    break
            if (step_count >= num_steps):
                break
        with torch.no_grad():
            # add log
            local_log = {}
            local_log = {'total_norm1': total_norm1, 'total_norm2': total_norm2} if 'clip' in kwargs.keys() and kwargs['clip'] > 0 else local_log
            return torch.nn.utils.parameters_to_vector(client_model.parameters()), local_log

    def local_update_epoch(self, client_model,data, epoch, batchsize):
        pass
    
    def evaluate_dataset(self, client_model, dataset, device):
        '''Evaluate on the given dataset'''
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        
        client_model.eval()
        self.global_model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)

                # client-side model forward
                activation = client_model(input)
                smashed_data = activation.clone().detach()

                # server-side model forward
                outputs = self.global_model(smashed_data)
                loss = self.criterion(outputs, target)
                acc1, acc5 = accuracy(outputs, target, topk=[1,5])
                losses.update(loss.item(), target.size(0))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))

            return losses, top1, top5,
            
class SplitMove():

    def __init__(self, client_num, select_num):
        self.client_num = client_num
        self.select_num = select_num
        self.initialize()
    
    def initialize(self):
        self.permutation = np.random.choice(range(self.client_num), self.client_num, replace=False)
        self.s = 0
    
    def move(self):
        if self.s + self.select_num > len(self.permutation):
            self.initialize()
        selected = self.permutation[self.s:self.s+self.select_num]
        self.s = self.s+self.select_num
        return selected

    