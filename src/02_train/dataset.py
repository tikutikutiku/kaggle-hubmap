import numpy as np
import pickle
import cv2
from os.path import join as opj
from torch.utils.data import Dataset
from transforms import get_transforms_train, get_transforms_valid
from utils import rle2mask

class HuBMAPDatasetTrain(Dataset):
    def __init__(self, df, config, mode='train'):
        self.data_df = df.copy().reset_index(drop=True)
        self.data_paths = self.data_df['data_path'].values
        self.filename_imgs = self.data_df['filename_img'].values
        self.filename_rles = self.data_df['filename_rle'].values
        self.config = config
        if mode=='train':
            self.transforms = get_transforms_train()
        else:
            self.transforms = get_transforms_valid()
        self.h, self.w = self.config['input_resolution']
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,idx):
        img_path = opj(self.data_paths[idx], self.filename_imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.w,self.h), interpolation=cv2.INTER_AREA)
        rle_path = opj(self.data_paths[idx], self.filename_rles[idx])
        with open(rle_path,'rb') as f:
            rle = pickle.load(f)
        mask = rle2mask(rle, shape=self.config['resolution'])
        mask = cv2.resize(mask, (self.w,self.h), interpolation=cv2.INTER_AREA)
        if self.transforms:
            augmented = self.transforms(image=img.astype(np.uint8), 
                                        mask=mask.astype(np.int8))
        img  = augmented['image']
        mask = augmented['mask']
        label = (mask.sum()>0).float()
        return {'img':img,'mask':mask,'label':label}