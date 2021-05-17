import numpy as np
import cv2
from torch.utils.data import Dataset
from os.path import join as opj
import rasterio
from rasterio.windows import Window

from transforms import get_transforms_test
from get_config import get_config
config = get_config()


class HuBMAPDatasetExternal(Dataset):
    def __init__(self, idx, filenames):
        super().__init__()
        filename = filenames[idx]
        path = opj(config['external_data_path'],filename)
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size'] # add to each input tile
        self.pred_sz = self.sz - 2*self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        self.transforms = get_transforms_test()
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3:
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        img = self.transforms(image=img)['image'] # to normalized tensor
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}
    
    
class HuBMAPDataset(Dataset):
    def __init__(self, idx, df, mode='test'):
        super().__init__()
        filename = df.loc[idx, 'id']+'.tiff'
        path = opj(config['INPUT_PATH'],mode,filename)
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size'] # add to each input tile
        self.pred_sz = self.sz - 2*self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        self.transforms = get_transforms_test()
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3:
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        img = self.transforms(image=img)['image'] # to normalized tensor
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}