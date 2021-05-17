import numpy as np
import cv2
import pickle
from os.path import join as opj
from utils import mask2rle, rle2mask

import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset


def generate_data(filename, i, img_patch, mask_patch, config):
    img_save_path = opj(config['OUTPUT_PATH'],filename+f'_img_{i:04d}.jpg')
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR) #rgb -> bgr
    cv2.imwrite(img_save_path, img_patch) #bgr -> rgb

    rle = mask2rle(mask_patch.squeeze(-1), mask_patch.shape[:2], small_mask_threshold=0)
    rle_save_path = opj(config['OUTPUT_PATH'],filename+f'_rle_{i:04d}')
    with open(rle_save_path, 'wb') as f:
        pickle.dump(rle, f)

    num_masked_pixels = mask_patch.sum()
    ratio_masked_area = mask_patch.sum() / (mask_patch.shape[0]*mask_patch.shape[1])
    std_img = img_patch.std()
    data = [img_save_path.split('/')[-1], rle_save_path.split('/')[-1], 
            num_masked_pixels, ratio_masked_area, std_img]
    return data
    

class HuBMAPDataset(Dataset):
    def __init__(self, df, filename, config, mode='train'):
        super().__init__()
        path = opj(config['INPUT_PATH'],mode,filename+'.tiff')
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.sz = config['tile_size']
        self.shift_h = config['shift_h']
        self.shift_w = config['shift_w']
        self.pad_h = self.sz - self.h % self.sz # add to whole slide
        self.pad_w = self.sz - self.w % self.sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz
        
        if self.h % self.sz < self.shift_h:
            self.num_h -= 1
        if self.w % self.sz < self.shift_w:
            self.num_w -= 1
        
        self.rle = df.loc[df['id']==filename, 'encoding'].values[0]
        self.mask = rle2mask(self.rle, shape=(self.h,self.w))
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.sz + self.shift_h
        x = i_w*self.sz + self.shift_w
        py0,py1 = max(0,y), min(y+self.sz, self.h)
        px0,px1 = max(0,x), min(x+self.sz, self.w)
        
        # placeholder for input tile (before resize)
        img_patch  = np.zeros((self.sz,self.sz,3), np.uint8)
        mask_patch = np.zeros((self.sz,self.sz), np.uint8)
        
        # replace the value for img patch
        if self.data.count == 3:
            img_patch[0:py1-py0, 0:px1-px0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((py0,py1),(px0,px1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img_patch[0:py1-py0, 0:px1-px0, i] =\
                    layer.read(1,window=Window.from_slices((py0,py1),(px0,px1)))
        
        # replace the value for mask patch
        mask_patch[0:py1-py0, 0:px1-px0] = self.mask[py0:py1,px0:px1]
        
        return {'img':img_patch, 'mask':mask_patch}
    
    
class HuBMAPDatasetExternal(Dataset):
    def __init__(self, df, filename, config):
        super().__init__()
        path = opj(config['external_data_path'],filename)
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.sz = config['tile_size']
        self.shift_h = config['shift_h']
        self.shift_w = config['shift_w']
        self.pad_h = self.sz - self.h % self.sz # add to whole slide
        self.pad_w = self.sz - self.w % self.sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz
        
        if self.h % self.sz < self.shift_h:
            self.num_h -= 1
        if self.w % self.sz < self.shift_w:
            self.num_w -= 1
        
        self.rle = df.loc[df['id']==filename, 'encoding'].values[0]       
        self.mask = rle2mask(self.rle, shape=(self.h,self.w))
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.sz + self.shift_h
        x = i_w*self.sz + self.shift_w
        py0,py1 = max(0,y), min(y+self.sz, self.h)
        px0,px1 = max(0,x), min(x+self.sz, self.w)
        
        # placeholder for input tile (before resize)
        img_patch  = np.zeros((self.sz,self.sz,3), np.uint8)
        mask_patch = np.zeros((self.sz,self.sz), np.uint8)
        
        # replace the value for img patch
        if self.data.count == 3:
            img_patch[0:py1-py0, 0:px1-px0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((py0,py1),(px0,px1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img_patch[0:py1-py0, 0:px1-px0, i] =\
                    layer.read(1,window=Window.from_slices((py0,py1),(px0,px1)))
        
        # replace the value for mask patch
        mask_patch[0:py1-py0, 0:px1-px0] = self.mask[py0:py1,px0:px1]
        
        return {'img':img_patch, 'mask':mask_patch}
    
    
def my_collate_fn(batch):
    img = []
    mask = []
    for sample in batch:
        img.append(sample['img'][None])
        mask.append(sample['mask'][None,:,:,None])
    img  = np.vstack(img)
    mask = np.vstack(mask)
    return {'img':img, 'mask':mask}