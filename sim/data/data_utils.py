import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class BaseDataset(Dataset):
    """Base dataset iterator, modified from
    https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/contrib/dataset/basic_dataset.py
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class FedDataset(object):
    def __init__(self, overallset, net_dataidx_map) -> None:
        self.map = net_dataidx_map
        self.num_datasets = len(net_dataidx_map)
        self.fedsetsizes = []
        self.fedsets = []
        self.totalsize = 0
        for i in range(len(net_dataidx_map)):
            # Note: For the choice between case 1 and case 2, please refer to (yipeng, 2023-11-14)
            # https://github.com/liyipeng00/convergence/tree/master#construct-local-datasets
            # Case 1
            self.fedsets.append(Subset(overallset, net_dataidx_map[i]))
            # Case 2
            #subset = [overallset[j] for j in net_dataidx_map[i]]
            #self.fedsets.append(BaseDataset(subset))
            
            self.fedsetsizes.append(len(net_dataidx_map[i]))
            self.totalsize += len(net_dataidx_map[i])
    
    def get_datasetsize(self, id):
        return self.fedsetsizes[id]

    def get_map(self, id):
        return self.map[id]

    def get_dataset(self, id):
        return self.fedsets[id]
    
    def get_subdataset(self, id, size):
        '''Get a subset of any local dataset, overallset > local dataset > sub local dataset'''
        subidx = np.random.choice(len(self.map[id]), size, replace=False)
        return Subset(self.fedsets[id], subidx)

    def get_dataloader(self, id, batch_size):
        """Get data loader"""
        subset = self.get_dataset(id)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)
    

'''Depreciate: since we found that `torch.utils.data.Subset` has the function [2023 06 18]'''
# class LocalDataset(Dataset):
#     r'''Generate Local dataset of the dataset given based on idxs.
#     Args:
#         dataset (torch.utils.data.Dataset): datasets
#         idx (list): A part indexes of the dataset given.
#     '''
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label