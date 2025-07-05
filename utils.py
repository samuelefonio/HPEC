
import torch
import pickle as pkl
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import Dataset
from cub2011 import Cub2011
from aircraft import Aircraft
from cars import StanfordCars
import numpy as np
import geoopt
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

DATASETS_CLASSES = {'mnist':10,
               'cifar10':10,
               'cifar100':100,
               'cub':200,
               'aircraft':100,
               'cars':196}

class SeparationLoss(nn.Module):
    """Large margin separation between hyperspherical protoypes, taken from https://github.com/VSainteuf/metric-guided-prototypes-pytorch"""

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos):
        """
        Args:
            protos (tensor): (N_prototypes x Embedding_dimension)
        """
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]
        ).to(protos.device)
        return M.max(dim=1)[0].mean()

def hyperspherical_embedding(dataset, device, embedding_dimension, seed):
    """
    Function to learn the prototypes according to the separationLoss Minimization
    embedding_dimension : 
    We use SGD as optimizer
    lr : learning rate
    momentum : momentum
    n_steps : number of steps for learning the prototypes
    wd : weight decay
    """
    lr=0.1
    momentum=0.9
    n_steps=1000
    wd=1e-4

    torch.manual_seed(seed) 
    mapping = torch.rand((DATASETS_CLASSES[dataset], embedding_dimension), device=device, requires_grad = True)
    
    optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)
    L_hp = SeparationLoss()
    
    for i in range(n_steps):
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        optimizer.zero_grad()
        loss = L_hp(mapping)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
    return mapping.detach()


def load_dataset(dataset_name, batch_size, num_workers=0, val=False):
    dataset_name = dataset_name.lower()
    print(f'loading the {dataset_name} dataset')
    if dataset_name == 'cifar100':
        return load_cifar100(batch_size, num_workers, val)
    elif dataset_name == 'cifar10':
        return load_cifar10(batch_size, num_workers, val)
    elif dataset_name == "mnist":
        return load_MNIST(batch_size, num_workers, val)
    elif dataset_name == "cub":
        return load_cub(batch_size, num_workers, val)
    elif dataset_name == "cars":
        return load_cars(batch_size, num_workers, val)
    elif dataset_name == "aircraft":
        return load_aircraft(batch_size, num_workers, val)
    else:
        raise Exception('Selected dataset is not available.')

def load_cifar100(batch_size, num_workers, val=False):

    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    size = transforms.Normalize(mean=mrgb, std=srgb)
    transformations=transforms.Compose([
                              transforms.RandomCrop(32, 4),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(15),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mrgb, std=srgb)])
    test_transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mrgb, std=srgb)])
    
    train = datasets.CIFAR100('./cifar100/', train = True, transform = transformations, download = True)
    test = datasets.CIFAR100('./cifar100/', train = False, transform = test_transformations, download = True)
    
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y
    def __len__(self):
        return len(self.data)

def load_cub(batch_size, num_workers, val=False):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train = Cub2011('./cub2011', train=True, transform=transform_train, download=True)
    test = Cub2011('./cub2011', train=False, transform = transform_test, download=True)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader

def load_aircraft(batch_size, num_workers, val=False):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train = Aircraft(root='./aircraft', train = True, transform=transform_train)
    test    = Aircraft(root='./aircraft', train = False, transform=transform_test)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader

def load_cars(batch_size, num_workers, val=False):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train = StanfordCars('./cars/', train = True, transform = transform_train)
    test = StanfordCars('./cars/', train = False, transform = transform_test)
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader
    
def load_MNIST(batch_size, num_workers=0, val=False):

    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=40, scale=(1.3,1.3)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.13066062],[0.30810776])
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.13066062],[0.30810776])
        ])
    train = datasets.MNIST('./MNIST', train=True, transform = transform_train, download=True)
    test = datasets.MNIST('./MNIST', train=False, transform = transform_test, download=True)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader

def load_cifar10(batch_size, num_workers=0, val=False):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = datasets.CIFAR10('./cifar10', train=True, transform = transform_train, download=True)
    test = datasets.CIFAR10('./cifar10', train=False, transform = transform_test, download=True)
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if val:
        valid_size = 0.20
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        validation = torch.utils.data.Subset(train, valid_indices)
        train = torch.utils.data.Subset(train, train_indices)
        validloader = data.DataLoader(validation, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    
    
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    if val: 
        return trainloader, testloader, validloader
    else:
        return trainloader, testloader


def load_optimizer(params, *args):
    optimname = args[0].lower()
    learning_rate = args[1]
    decay = args[2]
    momentum = args[3]
    
    if optimname == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    elif optimname == "adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "rsgd":
        optimizer = geoopt.optim.RiemannianSGD(params, lr=learning_rate, momentum = momentum, weight_decay=decay)
    elif optimname == "radam":
        optimizer = geoopt.optim.RiemannianAdam(params, lr=learning_rate,  weight_decay=decay)
    return optimizer


def double_clip(input_vector, kappa = 1, margin = 0.01, proto_r = 0.3, r = 1):
    input_norm = torch.norm(input_vector, dim = -1)
    shrinking_factor = 1/math.sqrt(kappa) * torch.tanh(math.sqrt(kappa) * input_norm/2)
    r_star = proto_r + shrinking_factor + margin
    max_norm = torch.clamp(torch.div(r_star,input_norm), min = r_star)
    output =  max_norm[:, None] * input_vector
    min_norm = torch.clamp(float(r)/input_norm, max = 1)
    return min_norm[:, None] * output

def backclip(input_vector, margin = 1, proto_r = 0.3):
    input_norm = torch.norm(input_vector, dim = -1)
    r_star = proto_r + margin
    max_norm = torch.clamp(torch.div(r_star,input_norm), min = r_star)
    return max_norm[:, None] * input_vector

def clip(input_vector, r):
    input_norm = torch.norm(input_vector, dim = -1)
    min_norm = torch.clamp(float(r)/input_norm, max = 1)
    return min_norm[:, None] * input_vector

def prediction(method, output, prototypes):
    if method in ['HPS']:
        output = nn.CosineSimilarity(dim=-1)(output[:,None,:], prototypes[None,:,:])
        pred = output.max(-1, keepdim=True)[1]
    elif method in ['CHPS','HBL','ECL','XE','NF']:
        pred = output.max(-1, keepdim=True)[1]
    return pred

class HPS_loss(nn.Module):
    def __init__(self, prototypes):
        super(HPS_loss, self).__init__()
        self.prototypes = prototypes

    def forward(self, output, target):
        dist = (1 - nn.CosineSimilarity(eps=1e-9)(output, self.prototypes[target])).pow(2).sum()
        return dist