import random
import os
import torch

VERSION = '05'

def get_config():
    config = {
        'split_seed_list':[0],
        'FOLD_LIST':[0,1,2,3],
        'VERSION':VERSION,
        'OUTPUT_PATH':f'./result/{VERSION}/',
        'INPUT_PATH':'../../input/hubmap-kidney-segmentation/',
        
        'train_data_path_list':[
            '../01_data_preparation/01_01/result/01_01/', 
            '../01_data_preparation/01_02/result/01_02/', 
        ],
        'pseudo_data_path_list':[
            '../04_data_preparation_pseudo_label/04_01_kaggle_data/result/04_01/',
            '../04_data_preparation_pseudo_label/04_02_kaggle_data_shift/result/04_02/',
        ],
        'external_pseudo_data_2_path_list':[
            '../04_data_preparation_pseudo_label/04_03_dataset_a_dib/result/04_03/',
            '../04_data_preparation_pseudo_label/04_04_dataset_a_dib_shift/result/04_04/',
        ],
        'external_pseudo_data_path_list':[
            '../04_data_preparation_pseudo_label/04_05_hubmap_external/result/04_05/',
            '../04_data_preparation_pseudo_label/04_06_hubmap_external_shift/result/04_06/',
        ],
        'pseudo_data_path_list_d488c759a':[
            '../04_data_preparation_pseudo_label/04_07_carno_zhao_label/result/04_07/',
            '../04_data_preparation_pseudo_label/04_08_carno_zhao_label_shift/result/04_08/',
        ],
        
        'model_name':'seresnext101', 
        'pretrain_path_list':None,
        'trn_idxs_list_path':None,
        'val_idxs_list_path':None,
        
        'num_classes':1,
        'input_resolution':(320,320),
        'resolution':(1024,1024),
        'dice_threshold':0.5,
        'small_mask_threshold':0, 
        'multiplier_bin':20, 
        'binned_max':4,
        'deepsupervision':True,
        'clfhead':True,
        'clf_threshold':None,
        'pad_size':256,
        
        'tta':1,
        'trn_batch_size':16,
        'test_batch_size':16, 
        
        'Adam':{
            'lr':1e-4, 
            'betas':(0.9, 0.999),
            'weight_decay':1e-5,
        },
        'SGD':{
            'lr':0.01,
            'momentum':0.9,
        },

        'lr_scheduler_name':'CosineAnnealingLR', #'OneCycleLR', #'ReduceLROnPlateau', #'StepLR',#'WarmUpLinearDecay', 

        'lr_scheduler':{
            'ReduceLROnPlateau':{
                'factor':0.8,
                'patience':5,
                'min_lr':1e-5,
                'verbose':True,
            },
            'OneCycleLR':{
                'pct_start':0.1,
                'div_factor':1e3, 
                'max_lr':1e-2,
                'epochs':25,
            },
            'CosineAnnealingLR':{
                'step_size_min':1e-6,
                't0':19,
                'tmult':1,
                'curr_epoch':-1,
                'last_epoch':-1,
            },
            'WarmUpLinearDecay':{
                'train_steps':40,
                'warm_up_step':3,
            },
            'StepLR':{
                'milestones':[1,2,3,20,40],
                'multipliers':[0.5,0.3,0.1,0.03,0.003],
            },
        },
        'snapshot':True,
        
        'restart_epoch_list':[1,1,1,1,1], 
        'unfreeze_epoch':1,
        'num_epochs':20,
        'early_stopping':True,
        'patience':50,

        #'FP16':True, always True
        'num_workers':4,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    return config