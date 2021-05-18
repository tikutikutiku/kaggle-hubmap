# kaggle-hubmap 1st place solution

This is the training code of the 1st place solution for Kaggle HuBMAP competition.

Solution summary : https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198  
Inference code : https://www.kaggle.com/tikutiku/hubmap-tilespadded-inference-v2?scriptVersionId=59475269  


#HARDWARE: (The following specs were used to create the original solution)  
Ubuntu 18.04 LTS (2TB boot disk)  
Core i7-10700K  
64GB memory  
1 x NVIDIA GeForce RTX3090  


#SOFTWARE: (python packages are detailed separately in `requirements.txt`)  
Python 3.7.3  
CUDA 11.2  
cuddn 8.1.1.33  
nvidia drivers v.460  


#Usage  
1. data preparation for kaggle train data  
```
cd src/01_data_preparation/01_01
python data_preparation_01_01.py

cd src/01_data_preparation/01_02
python data_preparation_01_02.py
```

2. train  
```
cd src/02_train
python train_02.py
```

3. pseudo-label generation  
```
cd src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data
python pseudo_label_kaggle_data_03_01.py

cd src/03_generate_pseudo_labels/03_02_pseudo_label_dataset_a_dib
python pseudo_label_kaggle_data_03_02.py

cd src/03_generate_pseudo_labels/03_03_pseudo_label_hubmap_external
python pseudo_label_hubmap_external_03_03.py
```

4. data preparation for pseudo-labeled data
```
cd src/04_data_preparation_pseudo_label/04_01_kaggle_data
python data_preparation_pseudo_label_04_01.py

cd src/04_data_preparation_pseudo_label/04_02_kaggle_data_shift
python data_preparation_pseudo_label_04_02.py

cd src/04_data_preparation_pseudo_label/04_03_dataset_a_dib
python data_preparation_pseudo_label_04_03.py

cd src/04_data_preparation_pseudo_label/04_04_dataset_a_dib_shift
python data_preparation_pseudo_label_04_04.py

cd src/04_data_preparation_pseudo_label/04_05_hubmap_external
python data_preparation_pseudo_label_04_05.py

cd src/04_data_preparation_pseudo_label/04_06_hubmap_external_shift
python data_preparation_pseudo_label_04_06.py

cd src/04_data_preparation_pseudo_label/04_07_carno_zhao_label
python data_preparation_pseudo_label_04_07.py

cd src/04_data_preparation_pseudo_label/04_08_carno_zhao_label_shift
python data_preparation_pseudo_label_04_08.py
```
6. train with kaggle train + pseudo-labeled data  
```
cd src/05_train_with_pseudo_labels
python train_05.py
```
