import importlib
import re

MODElNAMES = {
    'logistic': 'Logistic', 
    'mlp'     : 'MLP',
    'cnnmnist': 'CNNMnist',
    'lenet5'  : 'LeNet5',
    'cnncifar': 'CNNCifar'
    }
dataset_num_classes = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cinic10': 10, 'cifar100': 100, 'imagenet32': 1000}

def check(model, dataset):
    if 'resnet' in model:
        # model filter 1 if resnet
        model_pattern1 = r'logistic|mlp|lenet5|cnn|vgg|wvgg|resnetgnii|resnetii|wrngn|wrnnsgn|wrn|resnetgn\d+$|resnetgn\d+x\d+$'
        model_result1 = re.findall(model_pattern1, model)
        assert model_result1 != []
        model = re.sub(r'\d+', '', model)
    # model filter 2
    model_pattern2 = r'logistic|mlp|lenet5|cnn|vgg|wvgg|resnetii|resnetgnii|resnetgn|wrngn|wrnnsgn|wrn'    
    model_result = re.search(model_pattern2, model)
    model_result = model_result.group(0) if model_result != None else model_result
    
    if dataset == 'cinic10':
        dataset_result = 'cifar'
    else:
        dataset_pattern = r'mnist|cifar'
        dataset_result = re.search(dataset_pattern, dataset)
        dataset_result = dataset_result.group(0) if dataset_result != None else dataset_result
    return model_result, dataset_result


def build_model_split(model='lenet5', dataset='mnist', split=2):
    r'''build model for Split Learning based on model, dataset, cut layer. 
    All the options are below. 
    ResNetGN20 (ResNet20 with Group Norm)
    model           dataset             split layer     default
    ===========================================================
    MLP             mnist/fashionmnist   1*                1  
    LeNet5          mnist/fashionmnist   1,2*              2  
    CifarCNN        cifar10/100          1,2*              2
    Vgg11           cifar10/100          1,2,3,4*,5,6      4
    ResNetGN8       cifar10/100          1,2*              2
    WideResNetGN8   cifar10/100          1,2*              2
    ResNetGN20      cifar10/100          1,2*              2
    WideResNetGN20  cifar10/100          1,2*              2
    '''
    check_result = check(model, dataset)
    assert check_result[0] != None and check_result[1] != None

    # module
    if check_result[0] in ['mlp', 'lenet5', 'cnn']:
        module = importlib.import_module('.{}_{}'.format(check_result[0], check_result[1]), 'sim.models')
    elif check_result[0] in ['vgg', 'wvgg', 'resnetii', 'resnetgn', 'wrn', 'wrngn', 'wrnnsgn']:
        module = importlib.import_module('.{}_{}_split'.format(check_result[0], check_result[1]), 'sim.models')
    # class or function
    model_class = getattr(module, '{}_split'.format(model))
    # model
    num_classes = dataset_num_classes[dataset]
    client_model, server_model = model_class(num_classes=num_classes, split=split)
    return client_model, server_model


def build_model(model='lenet5', dataset='mnist'):
    '''https://github.com/lx10077/fedavgpy/blob/master/main.py (lines 100-103)'''
    check_result = check(model, dataset)
    assert check_result[0] != None and check_result[1] != None

    # module
    module = importlib.import_module('.{}_{}'.format(check_result[0], check_result[1]), 'sim.models')
    # class or function 
    if model in ['logistic', 'mlp', 'cnnmnist', 'lenet5', 'cnncifar']:
        model_class = getattr(module, MODElNAMES[model])
    else:
        model_class = getattr(module, model)
    # model
    num_classes = dataset_num_classes[dataset]
    if model in ['logistic', 'mlp', 'cnnmnist', 'lenet5'] and dataset in ['mnist', 'fashionmnist']:
        model = model_class()
    elif model == 'cnncifar' and dataset in ['cifar10', 'cifar100']:
        model = model_class(num_classes=num_classes)
    elif check_result[0] in ['vgg', 'wvgg', 'resnetii', 'resnetgnii', 'resnetgn', 'wrn', 'wrngn', 'wrnnsgn'] and dataset in ['cifar10', 'cifar100', 'cinic10']:
        model = model_class(num_classes=num_classes)
    return model
