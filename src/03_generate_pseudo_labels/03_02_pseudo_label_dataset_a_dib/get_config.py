import torch

VERSION = '03_02'

def get_config():
    config = {
        'VERSION':VERSION,
        'OUTPUT_PATH':f'./result/{VERSION}/',
        'INPUT_PATH':'../../../input/hubmap-kidney-segmentation/',
        'external_data_path':'../../../input/dataset_a_dib/',
        'split_seed_list':[0],
        'FOLD_LIST':[0,1,2,3], 
        'model_path':'../../02_train/result/02/',
        'model_name':'seresnext101',
        'val_idxs_list_path':'../../02_train/result/02/',

        'num_classes':1,
        'resolution':1024,
        'input_resolution':320,
        'deepsupervision':False, # always false for inference
        'clfhead':False,
        'clf_threshold':0.5,
        'small_mask_threshold':0, 
        'mask_threshold':0.5,
        'pad_size':256,

        'tta':4,
        'test_batch_size':12,

        'FP16':False,
        'num_workers':4,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    return config