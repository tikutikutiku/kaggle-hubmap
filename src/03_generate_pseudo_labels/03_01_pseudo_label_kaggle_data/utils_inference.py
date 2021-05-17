import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import math
from get_config import get_config
from dataset import HuBMAPDataset
from utils import mask2rle

config = get_config()
device = config['device']


def my_collate_fn(batch):
    img = []
    p = []
    q = []
    for sample in batch:
        img.append(sample['img'])
        p.append(sample['p'])
        q.append(sample['q'])
    img = torch.stack(img)
    return {'img':img, 'p':p, 'q':q}


seed = 0

def get_pred_mask(idx, df, model_list, mode='test'):
    ds = HuBMAPDataset(idx, df, mode)
    #rasterio cannot be used with multiple workers
    dl = DataLoader(ds,batch_size=config['test_batch_size'],
                    num_workers=0,shuffle=False,pin_memory=True,
                    collate_fn=my_collate_fn) 
    
    pred_mask = np.zeros((len(ds),ds.pred_sz,ds.pred_sz), dtype=np.uint8)
    
    i_data = 0
    for data in tqdm(dl):
        bs = data['img'].shape[0]
        img_patch = data['img'] # (bs,3,input_res,input_res)
        pred_mask_float = 0
        for model in model_list[seed]:
            with torch.no_grad():
                if config['tta']>0:
                    pred_mask_float += torch.sigmoid(model(img_patch.to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                if config['tta']>1:
                    # h-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,:,::-1]
                if config['tta']>2:
                    # v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,:]
                if config['tta']>3:
                    # h-v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1,-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,::-1]
        pred_mask_float = pred_mask_float / min(config['tta'],4) / len(model_list[seed]) # (bs,input_res,input_res)
        
        # resize
        pred_mask_float = np.vstack([cv2.resize(_mask.astype(np.float32), (ds.sz,ds.sz))[None] for _mask in pred_mask_float])
        
        # float to uint8
        pred_mask_int = (pred_mask_float>config['mask_threshold']).astype(np.uint8)
        
        # replace the values
        for j in range(bs):
            py0,py1,px0,px1 = data['p'][j]
            qy0,qy1,qx0,qx1 = data['q'][j]
            pred_mask[i_data+j,0:py1-py0, 0:px1-px0] = pred_mask_int[j, py0-qy0:py1-qy0, px0-qx0:px1-qx0] # (pred_sz,pred_sz)
        i_data += bs
    
    pred_mask = pred_mask.reshape(ds.num_h*ds.num_w, ds.pred_sz, ds.pred_sz).reshape(ds.num_h, ds.num_w, ds.pred_sz, ds.pred_sz)
    pred_mask = pred_mask.transpose(0,2,1,3).reshape(ds.num_h*ds.pred_sz, ds.num_w*ds.pred_sz)
    pred_mask = pred_mask[:ds.h,:ds.w] # back to the original slide size
    non_zero_ratio = (pred_mask).sum() / (ds.h*ds.w)
    print('non_zero_ratio = {:.4f}'.format(non_zero_ratio))
    return pred_mask,ds.h,ds.w

def get_rle(y_preds, h,w):
    rle = mask2rle(y_preds, shape=(h,w), small_mask_threshold=config['small_mask_threshold'])
    return rle