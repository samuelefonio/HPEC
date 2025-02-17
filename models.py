import torch.nn as nn
import torch
import resnet
import torchvision.models as torchmodel
import geoopt
from utils import *

DATASETS_CLASSES = {'mnist':10,
               'cifar10':10,
               'cifar100':100,
               'cub':200,
               'aircraft':100,
               'cars':196}

class SimpleCNN(nn.Module):
    '''Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch'''
    def __init__(self, output_dim = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16,32,3),nn.BatchNorm2d(32),nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 64),nn.Linear(64,output_dim))
        
    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = out.view(out.shape[0],out.shape[1], -1).max(-1)[0]
        out = self.fc(out)
        return out

def prototype_unify(num_classes):
    """
    Function that positionate the prototypes equidistant in 2 dimensions.
    """
    single_angle = 2 * math.pi / num_classes
    help_list = np.array(range(0, num_classes))
    angles = (help_list * single_angle).reshape(-1, 1)

    sin_points = np.sin(angles)
    cos_points = np.cos(angles)

    set_prototypes = torch.tensor(np.concatenate((cos_points, sin_points), axis=1))
    return set_prototypes
    
class Model(nn.Module):
    def __init__(
        self,
        model,
        device = 'cuda',
        dataset = 'mnist',
        output_dim = 8,
        temperature = 1.,
        clipping = 1,
        manifold = None,
        prototypes_ray = 0.3,
        margin = 0.1,
        seed = 0
    ):
        super(Model, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.output_dim = output_dim
        self.temperature = temperature
        self.clipping = clipping
        self.manifold = manifold
        self.prototypes_ray = prototypes_ray
        self.margin = margin
        
        self.seed = seed
        torch.manual_seed(seed)
        prototypes = hyperspherical_embedding(dataset, device, output_dim)
        prototypes = prototypes / torch.norm(prototypes, dim=1, keepdim=True) * self.prototypes_ray
        prototypes = self.manifold.expmap0(prototypes)
        self.prototypes = geoopt.ManifoldParameter(prototypes, manifold=manifold, requires_grad=False)

        
    def forward(self, images):
        embeddings = self.model(images)   
        embeddings = double_clip(embeddings, kappa = 1, margin = self.margin, proto_r = self.prototypes_ray)
        embeddings = self.manifold.expmap0(embeddings)
        return embeddings
    
def load_model(model, *args):
    method = args[0]
    args = args[1:]
    if method in ['HPS','XE']:
        return HPS_model(model, *args) 
    elif method == 'HBL':
        return HBL_model(model, *args)
    elif method == 'CHPS':
        return CHPS_model(model, *args)
    elif method == 'ECL':
        return ECL_model(model, *args)
    elif method == 'NF':
        return NF_model(model, *args)


class HPS_model(nn.Module):
    def __init__(self,
            model,
            device = 'cuda',
            dataset = 'mnist',
            output_dim = 8,
            temperature = 0.01,
            clipping = 1,
            manifold = None,
            seed = 42):
            super(HPS_model, self).__init__()
            self.model = model
            self.device = device
            self.output_dim = output_dim
            self.temperature = temperature
            self.dataset = dataset
            self.seed = seed
            self.prototypes = hyperspherical_embedding(self.dataset, self.device, self.output_dim, self.seed) # grad = False by default

    def forward(self, images):
        embeddings = self.model(images)      
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings / self.temperature
        
    
class HBL_model(nn.Module):
    def __init__(
        self,
        model,
        device = 'cuda',
        dataset = 'mnist',
        output_dim = 8,
        temperature = 0.01,
        clipping = 1,
        manifold = None,
        seed = 42
    ):
        super(HBL_model, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.output_dim = output_dim
        self.temperature = temperature
        self.clipping = clipping
        self.manifold = manifold
        
        prototypes = self.manifold.expmap0(-0.1 * torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device = self.device) + 0.1)
        self.prototypes = geoopt.ManifoldParameter(prototypes, manifold=manifold, requires_grad=True) #notice that in this method we have set a different optimizer for the prototypes
        
    def forward(self, images):
        embeddings = self.model(images)      
        embeddings = clip(embeddings, self.clipping)
        embeddings = self.manifold.expmap0(embeddings)
        dists = -self.manifold.dist(embeddings[:, None, :], self.prototypes[None, :, :])        
        return dists / self.temperature

class ECL_model(nn.Module):
    def __init__(
        self,
        model,
        device = 'cuda',
        dataset = 'mnist',
        output_dim = 8,
        temperature = 0.01,
        clipping = 1,
        manifold = None,
        seed = 42
    ):
        super(ECL_model, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.output_dim = output_dim
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device = self.device), requires_grad=True)

    def forward(self, images):
        embeddings = self.model(images)
        dists = -torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        return dists/self.temperature

class NF_model(nn.Module):
    def __init__(
        self,
        model,
        device = 'cuda',
        dataset = 'mnist',
        output_dim = 8,
        temperature = 0.01,
        clipping = 1,
        manifold = None,
        seed = 42
    ):
        super(NF_model, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.output_dim = output_dim
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device = self.device), requires_grad=True)

    def forward(self, images):
        embeddings = self.model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)
        dists = nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :])
        return dists/self.temperature

class CHPS_model(nn.Module):
    def __init__(
        self,
        model,
        device = 'cuda',
        dataset = 'mnist',
        output_dim = 8,
        temperature = 0.01,
        clipping = 1,
        manifold = None,
        seed = 42
    ):
        super(CHPS_model, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset
        self.output_dim = output_dim
        self.temperature = temperature
        self.seed = seed
        self.prototypes = hyperspherical_embedding(self.dataset, self.device, self.output_dim, self.seed) # grad = false by default
        
    def forward(self, images):
        embeddings = self.model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist = -(1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :]))     
        return dist/self.temperature


def load_backbone(dataset, output_dim):
    dataset = dataset.lower()
    if dataset == 'mnist':
        out_model = SimpleCNN(output_dim = output_dim)
    elif dataset == 'cifar10':
        out_model = resnet.resnet18()           
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    elif dataset == 'cifar100':
        out_model = resnet.resnet18()           
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    elif dataset == 'cub':
        out_model = torchmodel.resnet18(num_classes = 200, weights = None)            
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    elif dataset == 'aircraft':
        out_model = torchmodel.resnet18(num_classes = 100, weights = None)            
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    elif dataset == 'cars':
        out_model = torchmodel.resnet18(num_classes = 196, weights = None)            
        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, output_dim)
    else:
        raise Exception('Selected dataset is not available.')

    return out_model
