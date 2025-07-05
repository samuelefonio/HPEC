import os
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from PIL import Image
from torchvision.datasets.folder import default_loader


class StanfordCars(VisionDataset):
    """
    Code from a mix of https://github.com/pytorch/vision/issues/7545. Basically, there are many problems with this dataset. The original place where there were the data is closed.
    As a consequence, the data of these experiment are not from the official repository bt from the indications in the previous link. A sanity check has been done manually.
    """
    def __init__(self, root_path, train = True,  transform = None):
        if train:
            
            file_list = root_path+'/devkit/cars_train_annos.mat'
            root_path = root_path+'/cars_train'
        elif not train:
            
            file_list = root_path+'/devkit/cars_test_annos_withlabels.mat'
            root_path = root_path+'/cars_test'
        else:
            raise AttributeError('Please provide Train or test')
        self.root = root_path
        
        self.transform = transform
        self.loader = default_loader
        loaded_mat = sio.loadmat(file_list)
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            path = str(item[-1][0])
            label = int(item[-2][0]) - 1
            self.samples.append((path, label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)            
        if self.transform is not None:
            image = self.transform(image)
        return image, target
  

import numpy as np

if __name__ == '__main__':
    loaded_class = sio.loadmat('./cars/devkit/cars_meta.mat')['class_names'][0]
    loaded_mat = sio.loadmat('./cars/devkit/cars_test_annos_withlabels.mat')
    label = loaded_mat['annotations'][0][1][-2][0] - 1
