import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform_predict import *
import cv2


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, raster_numpy, transform=None):
        self.frame = raster_numpy
        self.transform = transform


    def __getitem__(self, idx):
        image_name = self.frame
        image = image_name.astype(np.uint8)        
        image = Image.fromarray(image)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return 1


def getTrainingData(batch_size=64,csv_data=''):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}


    csv = csv_data
    transformed_training_trans =  depthDataset(csv_file=csv,
                                        transform=transforms.Compose([
                                            #RandomHorizontalFlip(),
                                            CenterCrop([440, 440], [220, 220]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))
    #x = ConcatDataset([transformed_training1,transformed_training2,transformed_training3,transformed_training4,transformed_training5,transformed_training_no_trans])
    #dataloader_training = DataLoader(x,batch_size,
                                      #shuffle=True, num_workers=4, pin_memory=False)
    dataloader_training = DataLoader(transformed_training_trans, batch_size, num_workers=4, pin_memory=False)

   
   

    return dataloader_training


def getTestingData(batch_size=3, raster_numpy_or_torch=None):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
   

    # transformed_testing = depthDataset(csv_file='./data/building_test_meter_0.csv',
    #                                    transform=transforms.Compose([
    #                                        Scale(500),
    #                                        CenterCrop([400, 400],[400,400]),
    #                                        ToTensor(),
    #                                        Normalize(__imagenet_stats['mean'],
    #                                                  __imagenet_stats['std'])
    #                                    ]))
    img_win = raster_numpy_or_torch
    transformed_testing = depthDataset(raster_numpy=img_win,
                                       transform=transforms.Compose([
                                           CenterCrop([440, 440]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))
    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=12, pin_memory=False)
    return dataloader_testing


def getTestingData_1(batch_size=3, raster_numpy_or_torch=None):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}




    # img_win = raster_numpy_or_torch
    # transformed_testing =  depthDataset(raster_numpy=img_win,
    #                                         transform=transforms.Compose([
    #                                         #RandomHorizontalFlip(),
    #                                         CenterCrop([440, 440]),
    #                                         ToTensor(),
    #                                         Lighting(0.1, __imagenet_pca[
    #                                             'eigval'], __imagenet_pca['eigvec']),
    #                                         ColorJitter(
    #                                             brightness=0.4,
    #                                             contrast=0.4,
    #                                             saturation=0.4,
    #                                         ),
    #                                         Normalize(__imagenet_stats['mean'],
    #                                                   __imagenet_stats['std'])
    #                                     ]))


    img_win = raster_numpy_or_torch
    transformed_testing = depthDataset(raster_numpy=img_win,
                                       transform=transforms.Compose([
                                           CenterCrop([440, 440]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))
    
    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=12, pin_memory=False)
    return dataloader_testing