import torch
import time
import copy
import re
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image

from sim.data.partition import build_partition
from sim.models.build_models import build_model
from sim.utils.optim_utils import OptimKit, LrUpdater
from sim.utils.record_utils import logconfig, add_log, record_exp_result
from sim.utils.utils import setup_seed, AverageMeter, accuracy


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', default='vgg11', type=str, help='Model')
parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
parser.add_argument('-s', default=2, type=int, help='Index of split layer')
parser.add_argument('-R', default=200, type=int, help='Number of total training rounds')
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=100, type=int, help='Number of clients participate')
parser.add_argument('--partition', default='dir', type=str, choices=['dir', 'iid', 'exdir'], help='Data partition')
parser.add_argument('--alpha', default=10, type=float, nargs='*', help='The parameter `alpha` of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
parser.add_argument('--batch-size', default=20, type=int, help='Mini-batch size')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='Log, Log/Print')
parser.add_argument('--eval-num', default=1000, type=int, help='Number of evaluations')
parser.add_argument('--tail-eval-num', default=0, type=int, help='Evaluating the tail # rounds')
parser.add_argument('--device', default=0, type=int, help='Device')
parser.add_argument('--save-model', default=0, type=int, help='Whether to save model')
parser.add_argument('--start-round', default=0, type=int, help='Start')
parser.add_argument('--way', default=2, type=int, help='way')
args = parser.parse_args()

# nohup python main_fedavg.py -m mlp -d mnist -s 1 -R 100 -K 10 -M 500 -P 10 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &

torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
args.alpha = [int(args.alpha[0]), args.alpha[1]] if args.partition == 'exdir' else args.alpha
way = args.way
record_time = {
    'training_index': 0,
    'training_loadimg': 0,
    'training_transform': 0,
    'training_computation': 0,

    'inference1_index': 0,
    'inference1_loadimg': 0,
    'inference1_transform': 0,
    'inference1_computation': 0,

    'inference2_index': 0,
    'inference2_loadimg': 0,
    'inference2_transform': 0,
    'inference2_computation': 0,
}
accumulator = [0, 0, 0, 0, 0]

def sum_of_dict(dictionary):  
    total = 0  
    for key, value in dictionary.items():  
        total += value  
    return total 

def customize_record_name(args):
    '''FedAvg_M10_P10_K2_R4_mlp_mnist_exdir2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234_clip0.csv'''
    if args.partition == 'exdir':
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition == 'iid':
        partition = f'{args.partition}'
    record_name = f'FedAvg-VI{way}_M{args.M}_P{args.P}_K{args.K}_R{args.R}_{args.m}_{args.d}_{partition}'\
                + f'_{args.optim}{args.lr},{args.lr_decay},{args.momentum},{args.weight_decay}_b{args.batch_size}_seed{args.seed}_clip{args.clip}'
    return record_name
record_name = customize_record_name(args)

eval_batch_size = 20

###### CLIENT ######
class FedClient():
    global accumulator
    def __init__(self):
        pass
    
    def setup_criterion(self, criterion):
        self.criterion = criterion

    def setup_train_dataset(self, dataset):
        self.train_feddataset = dataset
    
    def setup_test_dataset(self, dataset):
        self.test_dataset = dataset

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit
    
    #client.local_update_step(model=copy.deepcopy(server.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, device=device, clip=args.clip)
    def local_update_step(self, c_id, model, num_steps, device, **kwargs):
        dataset=self.train_feddataset.get_dataset(c_id)
        data_loader = DataLoader(dataset, batch_size=self.optim_kit.batch_size, shuffle=True)
        optimizer = self.optim_kit.optim(model.parameters(), **self.optim_kit.settings)

        prev_model = copy.deepcopy(model)
        model.train()
        step_count = 0

        training_time = 0
        while(True):
            for input, target in data_loader:
                start_time_t3 = time.time()
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                optimizer.zero_grad()
                loss.backward()

                if 'clip' in kwargs.keys() and kwargs['clip'] > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=kwargs['clip'])

                optimizer.step()
                step_count += 1
                end_time_t3 = time.time()
                training_time = training_time + (end_time_t3 - start_time_t3)
                if (step_count >= num_steps):
                    break
            if (step_count >= num_steps):
                break

        accumulator[3] = accumulator[3] + training_time

        with torch.no_grad():
            curr_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            prev_vec = torch.nn.utils.parameters_to_vector(prev_model.parameters())
            delta_vec = curr_vec - prev_vec
            assert step_count == num_steps            
            # add log
            local_log = {}
            local_log = {'total_norm': total_norm} if 'clip' in kwargs.keys() and kwargs['clip'] > 0 else local_log
            return delta_vec, local_log

    def local_update_epoch(self, client_model,data, epoch, batchsize):
        pass

    def evaluate_dataset(self, model, dataset, device):
        '''Evaluate on the given dataset'''
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            training_time = 0
            for input, target in data_loader:
                start_time_t4 = time.time()
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=[1,5])
                losses.update(loss.item(), target.size(0))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))
                end_time_t4 = time.time()
                training_time = training_time + (end_time_t4 - start_time_t4)
            accumulator[4] = accumulator[4] + training_time
            return losses, top1, top5,

###### SERVER ######
class FedServer():
    def __init__(self):
        super(FedServer, self).__init__()
    
    def setup_model(self, model):
        self.global_model = model
    
    def setup_optim_settings(self, **settings):
        self.lr = settings['lr']
    
    def select_clients(self, num_clients, num_clients_per_round):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
        num_clients_per_round = min(num_clients_per_round, num_clients)
        return np.random.choice(num_clients, num_clients_per_round, replace=False)
    
    def global_update(self):
        with torch.no_grad():
            param_vec_curr = torch.nn.utils.parameters_to_vector(self.global_model.parameters()) + self.lr * self.delta_avg 
            return param_vec_curr
    
    def aggregate_reset(self):
        self.delta_avg = None
        self.weight_sum = torch.tensor(0) 
    
    def aggregate_update(self, local_delta, weight):
        with torch.no_grad():
            if self.delta_avg == None:
                self.delta_avg = torch.zeros_like(local_delta)
            self.delta_avg.add_(weight * local_delta)
            self.weight_sum.add_(weight)
    
    def aggregate_avg(self):
        with torch.no_grad():
            self.delta_avg.div_(self.weight_sum)

class CIFAR10T(CIFAR10):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        start_time_t0 = time.time()
        img, target = self.data[index], self.targets[index]
        end_time_t0 = time.time()
        accumulator[0] = accumulator[0] + (end_time_t0 - start_time_t0)

        start_time_t1 = time.time()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        end_time_t1 = time.time()
        accumulator[1] = accumulator[1] + (end_time_t1 - start_time_t1)

        start_time_t2 = time.time()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        end_time_t2 = time.time()
        accumulator[2] = accumulator[2] + (end_time_t2 - start_time_t2)

        return img, target

class BaseDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        start_time_t0 = time.time()
        img, target = self.dataset[index]
        end_time_t0 = time.time()
        accumulator[0] = accumulator[0] + (end_time_t0 - start_time_t0)

        start_time_t2 = time.time()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        end_time_t2 = time.time()
        accumulator[2] = accumulator[2] + (end_time_t2 - start_time_t2)
        
        return img, target

class FedDataset(object):
    def __init__(self, net_dataidx_map) -> None:
        global way
        self.map = net_dataidx_map
        self.num_datasets = len(net_dataidx_map)
        self.fedsetsizes = []
        self.fedsets = []
        self.totalsize = 0
        
        data_path = '../datasets/'
        cifar10_mean = [0.49139968, 0.48215827, 0.44653124]
        cifar10_std = [0.24703233, 0.24348505, 0.26158768]

        transform_train = transforms.Compose([
#            transforms.RandomCrop(32, padding=4),
#            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        # test test
        self.test_dataset = CIFAR10T(root=data_path, train=False, download=True, transform=transform_test)
        
        for i in range(len(net_dataidx_map)):
            self.fedsetsizes.append(len(net_dataidx_map[i]))
            self.totalsize += len(net_dataidx_map[i])

        if way == 1:
            self.train_dataset = CIFAR10T(root=data_path, train=True, download=True, transform=transform_train)
            for i in range(len(net_dataidx_map)):
                self.fedsets.append(Subset(self.train_dataset, net_dataidx_map[i]))
        elif way == 2:
            self.train_dataset = CIFAR10T(root=data_path, train=True, download=True, transform=transform_train)
            for i in range(len(net_dataidx_map)):
                subset = [self.train_dataset[j] for j in net_dataidx_map[i]]
                self.fedsets.append(BaseDataset(subset))
        elif way == 3:
            self.train_dataset = CIFAR10T(root=data_path, train=True, download=True, transform=None)
            for i in range(len(net_dataidx_map)):
                subset = [self.train_dataset[j] for j in net_dataidx_map[i]]
                self.fedsets.append(BaseDataset(dataset=subset, transform=transform_train))
    
    def get_datasetsize(self, id):
        return self.fedsetsizes[id]

    def get_map(self, id):
        return self.map[id]

    def get_dataset(self, id):
        return self.fedsets[id]
    
    def get_testdataset(self):
        return self.test_dataset

    def get_dataloader(self, id, batch_size):
        """Get data loader"""
        subset = self.get_dataset(id)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)

def main():
    global args, record_name, device, accumulator
    logconfig(name=record_name, flag=args.log)
    add_log('{}'.format(args), flag=args.log)
    add_log('record_name: {}'.format(record_name), flag=args.log)
    
    client = FedClient()
    server = FedServer()

    net_dataidx_map = build_partition(args.d, args.M, args.partition, [args.alpha[0], args.alpha[1]])
    feddataset = FedDataset(net_dataidx_map)
    client.setup_train_dataset(feddataset)
    client.setup_test_dataset(feddataset.get_testdataset())

    global_model = build_model(model=args.m, dataset=args.d)
    server.setup_model(global_model.to(device))
    add_log('{}'.format(global_model), flag=args.log)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())
    server.setup_optim_settings(lr=args.global_lr)
    
    # warm up
    if args.start_round != 0:
        warm_file = re.sub(r'R\d+', 'R{}'.format(args.start_round), record_name)
        warm_dict = torch.load('./save_model/{}.pt'.format(warm_file))
        warm_param_vec = warm_dict['model'].to(device)
        pointer = 0
        for param in server.global_model.parameters():
            num_param = param.numel()
            param.data = warm_param_vec[pointer:pointer + num_param].view_as(param).data
            pointer += num_param
        test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model, dataset=client.test_dataset, device=args.device)
        add_log("Round {}'s server test  acc: {:.2f}%, test  loss: {:.4f}".format(args.start_round, test_top1.avg, test_losses.avg), 'red', flag=args.log)

    start_time = time.time()
    add_log("Before reset: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(accumulator[0], accumulator[1], accumulator[2], accumulator[3], accumulator[4]), flag=args.log)
    accumulator = [0, 0, 0, 0, 0]
    accumulator_stamp = [0, 0, 0, 0, 0]
    record_exp_result(record_name, {'round':0})
    for round in range(args.start_round, args.R):
        server.aggregate_reset()
        selected_clients = server.select_clients(args.M, args.P)
        #add_log('selected clients: {}'.format(selected_clients), flag=args.log)
        for c_id in selected_clients:
            local_delta, local_update_log = client.local_update_step(c_id=c_id, model=copy.deepcopy(server.global_model), num_steps=args.K, device=device, clip=args.clip)
            #if local_update_log != {}:
            #    add_log('{}'.format(local_update_log.__str__()), flag=args.log) 
            server.aggregate_update(local_delta, weight=client.train_feddataset.get_datasetsize(c_id))
        server.aggregate_avg()
        param_vec_curr = server.global_update()
        torch.nn.utils.vector_to_parameters(param_vec_curr, server.global_model.parameters())

        client.optim_kit.update_lr()
        #add_log('lr={}'.format(client.optim_kit.settings['lr']), flag=args.log)
        
        record_time['training_index'] += accumulator[0]-accumulator_stamp[0]
        record_time['training_loadimg'] += accumulator[1]-accumulator_stamp[1]
        record_time['training_transform'] += accumulator[2]-accumulator_stamp[2]
        record_time['training_computation'] += accumulator[3]-accumulator_stamp[3]
        accumulator_stamp = accumulator.copy()
        #print("After training", accumulator)

        if (round+1) % max((args.R-args.start_round)//args.eval_num, 1) == 0 or (round+1) > args.R-args.tail_eval_num:
            # evaluate on train dataset (selected client)
            train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
            for c_id in selected_clients:
                local_losses, local_top1, local_top5 = \
                client.evaluate_dataset(model=server.global_model, dataset=client.train_feddataset.get_dataset(c_id), device=args.device)
                train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count), train_top5.update(local_top5.avg, local_top5.count)
            add_log("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
            
            record_time['inference1_index'] += accumulator[0]-accumulator_stamp[0]
            record_time['inference1_loadimg'] += accumulator[1]-accumulator_stamp[1]
            record_time['inference1_transform'] += accumulator[2]-accumulator_stamp[2]
            record_time['inference1_computation'] += accumulator[4]-accumulator_stamp[4]
            accumulator_stamp = accumulator.copy()
            add_log("After inference1: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(accumulator[0], accumulator[1], accumulator[2], accumulator[3], accumulator[4]), flag=args.log)

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model, dataset=client.test_dataset, device=args.device)
            add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)

            record_time['inference2_index'] += accumulator[0]-accumulator_stamp[0]
            record_time['inference2_loadimg'] += accumulator[1]-accumulator_stamp[1]
            record_time['inference2_transform'] += accumulator[2]-accumulator_stamp[2]
            record_time['inference2_computation'] += accumulator[4]-accumulator_stamp[4]
            accumulator_stamp = accumulator.copy()
            add_log("After inference2: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(accumulator[0], accumulator[1], accumulator[2], accumulator[3], accumulator[4]), flag=args.log)
            
            record_exp_result(record_name, {'round':round+1,
                             'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
                             'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg })

    if args.save_model == 1:
        torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())}, './save_model/{}.pt'.format(record_name))

    end_time = time.time()
    add_log("Record Time: |{:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| {:.2f}| = {:.2f}".format(\
        record_time['training_index'], record_time['training_loadimg'], record_time['training_transform'], record_time['training_computation'], \
        record_time['inference1_index'], record_time['inference1_loadimg'],record_time['inference1_transform'], record_time['inference1_computation'], \
        record_time['inference2_index'], record_time['inference2_loadimg'],record_time['inference2_transform'], record_time['inference2_computation'], \
        sum_of_dict(record_time)), flag=args.log)
    add_log("Total Training Time {:.2f}; Other Time {:.2f}".format(end_time - start_time, (end_time - start_time)-sum_of_dict(record_time)), flag=args.log)

if __name__ == '__main__':
    main()