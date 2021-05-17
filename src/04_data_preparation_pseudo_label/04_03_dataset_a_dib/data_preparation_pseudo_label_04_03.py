import sys
sys.path.insert(0, '../../')
import warnings
warnings.simplefilter('ignore')
from utils import fix_seed
from utils_data_generation import HuBMAPDatasetExternal, my_collate_fn, generate_data

import random
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
from os.path import join as opj
import gc


VERSION = '04_03'

def get_config():
    config = {
        'VERSION':VERSION,
        'OUTPUT_PATH':f'./result/{VERSION}/',
        'INPUT_PATH':'../../../input/hubmap-kidney-segmentation/',
        'external_data_path':'../../../input/dataset_a_dib/',
        'pseudo_label_path':'../../03_generate_pseudo_labels/03_02_pseudo_label_dataset_a_dib/result/03_02/external_dataset_a_dib.csv',
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'tile_size':1024,
        'batch_size':16,
        'shift_h':0,
        'shift_w':0,
    }
    return config
    
if __name__=='__main__':
    # config
    fix_seed(2021)
    config = get_config()
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)

    # import pseudo-label
    pseudo_df = pd.read_csv(config['pseudo_label_path'])
    pseudo_df = pseudo_df[['filename','predicted']].copy().rename(columns={'filename':'id','predicted':'encoding'})
    print(pseudo_df.shape)
    
    # generate pseudo-labeled data
    data_list = []
    for idx,filename in enumerate(pseudo_df['id'].values):
        print('idx = {}, {}'.format(idx,filename))
        ds = HuBMAPDatasetExternal(pseudo_df, filename, config)

        #rasterio cannot be used with multiple workers
        dl = DataLoader(ds,batch_size=config['batch_size'],
                        num_workers=0,shuffle=False,pin_memory=True,
                        collate_fn=my_collate_fn)
        img_patches  = []
        mask_patches = []
        for data in tqdm(dl):
            img_patch = data['img']
            mask_patch = data['mask']
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
        img_patches  = np.vstack(img_patches)
        mask_patches = np.vstack(mask_patches)

        # sort by number of masked pixels
        bs,sz,sz,c = img_patches.shape
        idxs = np.argsort(mask_patches.reshape(bs,-1).sum(axis=1))[::-1]
        img_patches  = img_patches[idxs].reshape(-1,sz,sz,c)
        mask_patches = mask_patches[idxs].reshape(-1,sz,sz,1)

        data = Parallel(n_jobs=-1)(delayed(generate_data)(filename, i, x, y, config) for i,(x,y) in enumerate(zip(img_patches, mask_patches))) 
        data_list.append(data)
     
    # save
    data_df = pd.concat([pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0).reset_index(drop=True)
    data_df.columns = ['filename_img', 'filename_rle', 'num_masked_pixels', 'ratio_masked_area', 'std_img']
    print(data_df.shape)
    data_df.to_csv(opj(OUTPUT_PATH, 'data.csv'), index=False)