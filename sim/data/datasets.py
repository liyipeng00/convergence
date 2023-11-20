from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torchvision import datasets

def build_dataset(dataset_name='mnist', dataset_dir = '../datasets/'):
    if dataset_name == 'mnist':
        train_dataset, test_dataset = dataset_mnist(dataset_dir)
    elif dataset_name == 'fashionmnist':
        train_dataset, test_dataset = dataset_fashionmnist(dataset_dir)
    elif dataset_name == 'cifar10':
        train_dataset, test_dataset = dataset_cifar10(dataset_dir)
    elif dataset_name == 'cifar100':
        train_dataset, test_dataset = dataset_cifar100(dataset_dir)
    elif dataset_name == 'cinic10':
        train_dataset, test_dataset = dataset_cinic10(dataset_dir)
    return train_dataset, test_dataset


def dataset_mnist(data_path):
    '''
    https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
    '''
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) # mean: 0.13066235184669495, std:0.30810782313346863
    train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def dataset_fashionmnist(data_path):
    '''
    Replace the mean and std with the data generated manually
    https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,), (0.3530,))])
    
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,), (0.3530,)),
                        ])

    train_dataset = FashionMNIST(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = FashionMNIST(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar10(data_path):
    '''
    https://github.com/JYWa/FedNova/blob/master/util_v4.py
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar10/data_loader.py
    '''
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar100(data_path):
    '''
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar100/data_loader.py
    '''
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def dataset_cinic10(data_path):
    '''
    https://github.com/BayesWatch/cinic-10
    https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    train_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/train/'), transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    test_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/test/'), transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    for i in range(0, 3):
        # to judge if the sample sequence is the same at different times
        train_dataset, test_dataset = dataset_mnist('../datasets/')
        print(train_dataset.targets[:30])
   
    
    
