import random
import torch
import numpy as np
import os
import time
from os.path import isfile
from os.path import join as opj

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    
def elapsed_time(start_time):
    return time.time() - start_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rle2mask(rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    Returns numpy array <- 1(mask), 0(background)
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


def mask2rle(img, shape, small_mask_threshold):
    '''
    Convert mask to rle.
    img: numpy array <- 1(mask), 0(background)
    Returns run length as string formated
    
    pixels = np.array([1,1,1,0,0,1,0,1,1]) #-> rle = '1 3 6 1 8 2'
    pixels = np.concatenate([[0], pixels, [0]]) #[0,1,1,1,0,0,1,0,1,1,0]
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 #[ 1  4  6  7  8 10] bit change points
    print(runs[1::2]) #[4 7 10]
    print(runs[::2]) #[1 6 8]
    runs[1::2] -= runs[::2]
    print(runs) #[1 3 6 1 8 2]
    '''
    if img.shape[:2] != shape:
        h,w = shape
        img = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.int8) 
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs[1::2].sum() <= small_mask_threshold:
        return ''
    else:
        return ' '.join(str(x) for x in runs)