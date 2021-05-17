import torch

VERSION = '03_03'

mapper = {
    'train':{ # 8 in total
        'VAN0003-LK-33-2-PAS_FFPE.ome.tif':'aaa6a05cc',  # train
        'VAN0006-LK-2-85-PAS_registered.ome.tif':'54f2eec69', # train
        'VAN0006-LK-7-2-PAS_FFPE.ome.tif':'cb2d976f4', # train
        'VAN0008-RK-403-100-PAS_registered.ome.tif':'1e2425f28', # train
        'VAN0009-LK-102-7-PAS_registered.ome.tif':'e79de561c', # train
        
        'VAN0011-RK-8-2-PAS_FFPE.ome.tif':'0486052bb', # train
        'VAN0012-RK-103-75-PAS_registered.ome.tif':'095bf7a1f', # train
        'VAN0014-LK-207-2-PAS_FFPE.ome.tif':'2f6ecfcdf', # train
    },
    'public':{ # 5 in total
        'VAN0005-RK-8-2-PAS_FFPE.ome.tif':'b9a3865fc', # public
        'VAN0005-RK-4-172-PAS_registered.ome.tif':'afa5e8098', # public
        'VAN0009-LK-106-2-PAS_FFPE.ome.tif':'b2dc8411c', # public
        'VAN0010-LK-155-40-PAS_registered.ome.tif':'c68fe75ea', # public
        'VAN0013-LK-202-96-PAS_registered.ome.tif':'26dc41664', # public
    },
    'new_train':{ # 2 in total
        'VAN0016-LK-202-89-PAS_registered.ome.tif':'4ef6695ce', # new-train
        'VAN0016-LK-208-2-PAS_FFPE.ome.tif':'8242609fa', # new-train
    },
    'new_public':{ # 5 in totall
        'VAN0005-RK-1-1-PAS_registered.ome.tif':'aa05346ff', # new-public
        'VAN0007-LK-203-103-PAS_registered.ome.tif':'d488c759a', # new-public 
        'VAN0010-LK-160-2-PAS_FFPE.ome.tif':'2ec3f1bb9', # new-public
        'VAN0013-LK-206-2-PAS_FFPE.ome.tif':'3589adb90', # new-public
        'VAN0014-LK-203-108-PAS_registered.ome.tif':'57512b7f1', # new-public
    },
    'external':{ # 2 in total
        'VAN0003-LK-32-21-PAS_registered.ome.tif':None, 
        'VAN0011-RK-3-10-PAS_registered.ome.tif':None,
    },
}

def get_config():
    config = {
        'VERSION':VERSION,
        'OUTPUT_PATH':f'./result/{VERSION}/',
        'INPUT_PATH':'../../../input/hubmap-kidney-segmentation/',
        'external_data_path':'../../../input/hubmap-external/', 
        'filename_external':mapper['external'],
        
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