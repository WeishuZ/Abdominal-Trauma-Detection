# RSNA 2023 Abdominal Trauma Detection Competition

[RSNA 2023 Abdominal Trauma Detection Competition](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview)

------

## Introduction

Hardware:

- CPU : `13th Gen Intel(R) Core(TM) i7-13700K 8 cores 24 threads`
- memory : `64GB`
- GPU : `NVIDIA GeForce RTX 3090 GPU with 32 GB VRAM`
- number of GPUs : `1`

OS/platform : `Windows 11`


------

## Prerequisites

- Clone the repository

- Setup the environment:
  `pip install -r requirements.txt`

- Download the data in the `data` folder:

  - download [Competition data](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data) and change folder name to `rsna-2023-abdominal-trauma-detection`

    ```
    data/rsna-2023-abdominal-trauma-detection
    ├── image_level_labels.csv
    ├── sample_submission.csv
    ├── segmentations [206 entries exceeds filelimit, not opening dir]
    ├── test_dicom_tags.parquet
    ├── test_images
    ├── test_series_meta.csv
    ├── train.csv
    ├── train_dicom_tags.parquet
    ├── train_images
    └── train_series_meta.csv
    ```

## Run The pipeline

### Train model

#### 1. Train 3D segmentation model

1. `cd src/Segmentation` : to run segmentation file

2. `python train.py` : to train segmentation model

3. `python process_data.py` : generate classification data based on segmentation model

   - segmentation model folder

     ```
     ├── results
     │   ├── models
     │   │   ├── segmentations
     │   │   │    └── train
     ```
   
   - segmentation logs folder
   
     ```
     ├── results
     │   ├── logs
     │   │   ├── segmentations
     │   │   │    └── train
     ```
     
   - data structure of segmented output will be :
   
     ```
     ├── results               
     │   ├── data
     │   │   ├── segmentation
     │   │   │   ├── data        
     │   │   │   │   ├── liver
     │   │   │   │   ├── kidney
     │   │   │   │   ├── spleen
     │   │   │   │   ├── bowel
     ```
     

#### 2. Train Bowel Model

1. `cd src/Classification_bowel` : to run bowel classification file

2. `python train.py` : to train bowel model

   - bowel model folder

     ```
     ├── results
     │   ├── models
     │   │   ├── bowel
     │   │   │    └── train
     ```

   - train bowel logs folder

     ```
     ├── results
     │   ├── logs
     │   │   ├── bowel
     │   │   │    └── train
     ```

#### 3. Train Kidney Model

1. `cd src/Classification_kidney `: to run kidney classification file

2. `python train.py` : to train kidney model

   - kidney model folder

     ```
     ├── results
     │   ├── models
     │   │   ├── kidney
     │   │   │    └── train
     ```

   - train bowel logs folder

     ```
     ├── results
     │   ├── logs
     │   │   ├── kidney
     │   │   │    └── train
     ```

#### 4. Train Liver Model

1. `cd src/Classification_liver `: to run liver classification file

2. `python train.py` : to train liver model

   - liver model folder

     ```
     ├── results
     │   ├── models
     │   │   ├── liver
     │   │   │    └── train
     ```

   - liver logs folder

     ```
     ├── results
     │   ├── logs
     │   │   ├── liver
     │   │   │    └── train
     ```

#### 5. Train Spleen Model

1. `cd src/Classification_spleen `: to run spleen classification file

2. `python train.py` : to train spleen model

   - spleen model folder

     ```
     ├── results
     │   ├── models
     │   │   ├── liver
     │   │   │    └── train
     ```

   - spleen logs folder

     ```
     ├── results
     │   ├── logs
     │   │   ├── liver
     │   │   │    └── train
     ```

#### 6. Train all Model

1. `cd bash` : to run bash file
2. `sh train_all.sh` : to train all model
