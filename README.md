# kaggle-hubmap 1st place solution

This is the training code of the 1st place solution for Kaggle HuBMAP competition.

Solution summary : https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198  
Inference code : https://www.kaggle.com/tikutiku/hubmap-tilespadded-inference-v2?scriptVersionId=59475269  

Usage  
1. data preparation for kaggle train data  
```
cd src/01_data_preparation/01_01
python data_preparation_01_01.py

cd src/01_data_preparation/01_02
python data_preparation_01_02.py
```
2. train  
3. pseudo-label generation  
4. data preparation for pseudo-labeled data  
5. train with kaggle train + pseudo-labeled data  
