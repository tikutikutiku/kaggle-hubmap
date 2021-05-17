import numpy as np

def dice_sum(img, mask, dice_threshold, small_mask_threshold):
    batch = img.shape[0]
    
    #flatten
    img  = img.reshape(batch,1,-1)
    mask = mask.reshape(batch,1,-1)
    
    dice_array = np.zeros((batch,1))
    for i in range(batch):
        for j in range(1):
            img_ij   = (img[i,j,:]>dice_threshold).astype(np.float32)
            mask_ij  = mask[i,j,:]
            if (np.sum(img_ij)<=small_mask_threshold):
                img_ij = np.zeros_like(img_ij)
            
            if (np.sum(img_ij)<=small_mask_threshold)&(np.sum(mask_ij)==0):
                dice_array[i][j] = 1.0
            else:
                dice_array[i][j] = 2*np.sum(img_ij*mask_ij) / (np.sum(img_ij) + np.sum(mask_ij) + 1e-12)
    return dice_array.sum()


def dice_sum_2(img, mask, dice_threshold):
    batch = img.shape[0]
    
    #flatten
    img  = img.reshape(batch,1,-1)
    mask = mask.reshape(batch,1,-1)
    
    dice_numerator = 0
    dice_denominator = 0
    for i in range(batch):
        for j in range(1):
            img_ij  = (img[i,j,:]>dice_threshold).astype(np.float32)
            mask_ij = mask[i,j,:]
            dice_numerator   += 2*np.sum(img_ij*mask_ij)
            dice_denominator += np.sum(img_ij) + np.sum(mask_ij)
    return dice_numerator, dice_denominator