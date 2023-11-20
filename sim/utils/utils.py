import copy
import random
import numpy as np
import torch

class AverageMeter(object):
    r"""Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    r"""Computes the precision@k for the specified values of k.
    for example, topk(1,2,3) -> [tensor([33.3333]), tensor([66.6667]), tensor([66.6667])]
    Ref: https://github.com/JYWa/FedNova/blob/47b4e096dfb19dc43c728896fff335a5befb645d/util_v4.py#L213
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
                
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
                            
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) # 100, 0.88 -> 88
        return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, args):
    r"""Decay the learning rate based on schedule.
    One param_group for one model.
    Args: 
        args.schedule (list): e.g. [60, 80]
    """
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def average_weights(w, datasize):
    r"""Returns the average of the weights.
    Args:
        datasize (list): the datasize of all local datasets.
    """   
    datasize = torch.tensor(datasize)
    for i, data in enumerate(datasize):
        for key in w[i].keys():
            # if we use "w[i][key] *= float(data)", when resnet parameters may have different type that is not float
            w[i][key] *= data.type_as(w[i][key])
    
    w_avg = copy.deepcopy(w[0])
   
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg

def paramsum(params):
    r'''get the difference(or gap) of two models(or layers))'s parameters'''
    params = list(params)
    with torch.no_grad():
        result = sum(x.data.sum() for x in params)
    print(result.item())
    return result.item()

def modelsgap(params1, params2):
    r'''get the difference(or gap) of two models(or layers))'s parameters'''
    with torch.no_grad():
        gap = sum((x.data - y.data).pow(2.0).sum() for x, y in zip(params1, params2))
    return gap.item()
    
def param_gradsum(params, i):
    a = []
    for p in params:
        if p.requires_grad == True and p.grad != None:
            a.append(p.grad.data.sum())
    b = sum(a)
    print(b, i)
    return b

# from collections import defaultdict
# def group_clients(clients_num=20, servers_num=2, order:bool =True) -> defaultdict:
#     r'''Group clients associated with servers.
#     If order is bool, return ordered division of clients in the group: group1 client[0] to client[x];
#     else, return random division.
#     Args:
#         order: True means clients are grouped in order, False means random selection.
#     Examples:
#         >>> from collections import defaultdict
#         >>> import random
#         >>> dict = group_clients(6, 3, 1)
#         >>> defaultdict(<class 'list'>, {0: [0, 1], 1: [2, 3], 2: [4, 5]})
#     '''
#     group_clients_dict = defaultdict(list)
#     clients_list = [i for i in range(0, clients_num)]
    
#     if not order:
#         clients_list = random.sample(clients_list, clients_num)
#     start = 0
#     step = int(clients_num/servers_num)
#     diff = clients_num - step * servers_num   
#     if diff != 0:
#         end = step + 1
#         diff -= 1
#     else:
#         end = step
    
#     for n in range(servers_num):
#         clients_sublist = clients_list[start:end]
#         group_clients_dict[n]= clients_sublist           
#         if end+step <= clients_num:
#             start = end  
#             if diff != 0:
#                 end += step + 1
#                 diff -= 1
#             else:
#                 end += step        
#         else:
#             start = end
#             end = clients_num

#     return group_clients_dict

if __name__ == '__main__':
    pass
    


