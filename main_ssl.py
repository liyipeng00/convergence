''' Sequential Split Learning
Refs:
[1] https://github.com/chandra2thapa/SplitFed
[2] https://github.com/garrisongys/SplitFed
[3] https://github.com/AshwinRJ/Federated-Learning-PyTorch
[4] https://github.com/Divyansh03/FedExP
[2022 08 03]'''
import torch
import time
import copy
import re

from sim.algorithms.splitbase import SplitClient, SplitServer, SplitMove
from sim.data.data_utils import FedDataset
from sim.data.datasets import build_dataset
from sim.data.partition import build_partition
from sim.models.build_models import build_model_split
from sim.utils.record_utils import logconfig, add_log, record_exp_result2
from sim.utils.utils import setup_seed, AverageMeter
from sim.utils.optim_utils import OptimKit, LrUpdater

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
parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--batch-size', default=50, type=int, help='Mini-batch size')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='Log, Log/Print')
parser.add_argument('--eval-num', default=1000, type=int, help='Number of evaluations')
parser.add_argument('--tail-eval-num', default=0, type=int, help='Evaluating the tail # rounds')
parser.add_argument('--device', default=0, type=int, help='Device')
parser.add_argument('--save-model', default=0, type=int, help='Whether to save model')
parser.add_argument('--start-round', default=0, type=int, help='Start')
args = parser.parse_args()

# nohup python main_ssl.py -m mlp -d mnist -s 1 -R 100 -K 10 -M 500 -P 10 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &

torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
args.alpha = [int(args.alpha[0]), args.alpha[1]] if args.partition == 'exdir' else args.alpha

def customize_record_name(args):
    '''SSL-III_M10_P10_K2_R4_mlp-1_mnist_exdir2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234_clip0.csv'''
    if args.partition == 'exdir':
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition == 'iid':
        partition = f'{args.partition}'
    record_name = f'SSL-III_M{args.M}_P{args.P}_K{args.K}_R{args.R}_{args.m}-{args.s}_{args.d}_{partition}'\
                + f'_{args.optim}{args.lr},{args.lr_decay},{args.momentum},{args.weight_decay}_b{args.batch_size}_seed{args.seed}_clip{args.clip}'
    return record_name
record_name = customize_record_name(args)

def main():
    global args, record_name, device
    logconfig(name=record_name, flag=args.log)
    add_log('{}'.format(args), flag=args.log)
    add_log('record_name: {}'.format(record_name), flag=args.log)
    
    client = SplitClient()
    server = SplitServer()

    train_dataset, test_dataset = build_dataset(args.d)
    net_dataidx_map = build_partition(args.d, args.M, args.partition, [args.alpha[0], args.alpha[1]])
    train_feddataset = FedDataset(train_dataset, net_dataidx_map)
    client.setup_train_dataset(train_feddataset)
    client.setup_test_dataset(test_dataset)

    client_global_model, server_global_model = build_model_split(model=args.m, dataset=args.d, split=args.s)
    client.setup_model(client_global_model.to(device))
    server.setup_model(server_global_model.to(device))
    add_log('{}\n{}'.format(client_global_model, server_global_model), flag=args.log)

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    server_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    server_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    client.setup_optim_kit(client_optim_kit)
    server.setup_optim_kit(server_optim_kit)
    server.setup_criterion(torch.nn.CrossEntropyLoss())

    start_time = time.time()
    # warm up
    if args.start_round != 0:
        warm_file = re.sub(r'R\d+', 'R{}'.format(args.start_round), record_name)
        warm_dict = torch.load('./save_model/{}.pt'.format(warm_file))
        warm_param_vec = warm_dict['model'].to(device)
        pointer = 0
        for param in client.global_model.parameters():
            num_param = param.numel()
            param.data = warm_param_vec[pointer:pointer + num_param].view_as(param).data
            pointer += num_param
        for param in server.global_model.parameters():
            num_param = param.numel()
            param.data = warm_param_vec[pointer:pointer + num_param].view_as(param).data
            pointer += num_param
        test_losses, test_top1, test_top5 = server.evaluate_dataset(client_model=client.global_model, dataset=client.test_dataset, device=args.device)
        add_log("Round {}'s server test  acc: {:.2f}%, test  loss: {:.4f}".format(args.start_round, test_top1.avg, test_losses.avg), 'red', flag=args.log)
    
    record_exp_result2(record_name, {'round':0})
    for round in range(args.start_round, args.R):
        selected_clients = server.select_clients(args.M, args.P)
        add_log('selected clients: {}'.format(selected_clients), flag=args.log)
        for c_id in selected_clients:
            client_local_param, local_update_log = server.local_update_step(client_model=copy.deepcopy(client.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, client_optim_kit=client_optim_kit, device=device, clip=args.clip)
            # if local_update_log != {}:
            #     add_log('{}'.format(local_update_log.__str__()), flag=args.log)
            torch.nn.utils.vector_to_parameters(client_local_param, client.global_model.parameters())
        
        client.optim_kit.update_lr()
        server.optim_kit.update_lr()
        add_log(f"client lr={client.optim_kit.settings['lr']}, server lr={server.optim_kit.settings['lr']}", flag=args.log)

        if (round+1) % max((args.R-args.start_round)//args.eval_num, 1) == 0 or (round+1) > args.R-args.tail_eval_num:
            # evaluate on train dataset
            train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
            for c_id in selected_clients:
                local_losses, local_top1, local_top5 = \
                server.evaluate_dataset(client_model=client.global_model, dataset=client.train_feddataset.get_dataset(c_id), device=args.device)
                train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count), train_top5.update(local_top5.avg, local_top5.count)
            add_log("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
            
            # evaludate on train dataset (random client)
            train2_losses, train2_top1, train2_top5 = AverageMeter(), AverageMeter(), AverageMeter()
            rand_eval_clients = server.select_clients(args.M, args.P)
            for c_id in rand_eval_clients:
                local_losses, local_top1, local_top5 = \
                server.evaluate_dataset(client_model=client.global_model, dataset=client.train_feddataset.get_dataset(c_id), device=args.device)
                train2_losses.update(local_losses.avg, local_losses.count), train2_top1.update(local_top1.avg, local_top1.count), train2_top5.update(local_top5.avg, local_top5.count)
            add_log("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg), 'blue', flag=args.log)

            # evaluate on test dataset
            test_losses, test_top1, test_top5 = server.evaluate_dataset(client_model=client.global_model, dataset=client.test_dataset, device=args.device)
            add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)

            record_exp_result2(record_name, {'round':round+1,
                             'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
                             'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
                             'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg })
    if args.save_model == 1:
        torch.save({'model': torch.cat((torch.nn.utils.parameters_to_vector(client.global_model.parameters()),
                    + torch.nn.utils.parameters_to_vector(server.global_model.parameters())))}, './save_model/{}.pt'.format(record_name))
    
    end_time = time.time()
    add_log("TrainingTime: {} sec".format(end_time - start_time), flag=args.log)

if __name__ == '__main__':
    main()