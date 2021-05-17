import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations.pytorch import ToTensor
from get_config import *
config = get_config()

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_train():
    transform_train = Compose([
        #Basic
        RandomRotate90(p=1),
        HorizontalFlip(p=0.5),
        
        #Morphology
        ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
        GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        GaussianBlur(blur_limit=(3,7), p=0.5),
        
        #Color
        RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                 brightness_by_max=True,p=0.5),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
                           val_shift_limit=0, p=0.5),
        
        CoarseDropout(max_holes=2, 
                      max_height=config['input_resolution'][0]//4, max_width=config['input_resolution'][1]//4, 
                      min_holes=1,
                      min_height=config['input_resolution'][0]//16, min_width=config['input_resolution'][1]//16, 
                      fill_value=0, mask_fill_value=0, p=0.5),
        
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensor(),
    ])
    return transform_train


def get_transforms_valid():
    transform_valid = Compose([
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensor(),
    ] )
    return transform_valid


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean