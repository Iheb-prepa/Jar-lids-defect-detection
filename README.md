# Defect Detection in Jar Lids Using Image Anomaly Detection
![p62_bbox_551](https://github.com/user-attachments/assets/0907cd8e-2b34-48a7-873f-9ef817928344)

## Overview
This project implements an image anomaly detection model for identifying defects in jar lids using Anomalib, a library specifically designed for anomaly detection in images. 

Note: the dataset is available in Kaggle but is not intended for commercial use, however, the code in this repo is open-source and you can use it for your needs.

Source of the dataset: https://www.kaggle.com/datasets/rrighart/jarlids.

## Data preparation
The dataset contains 168 image files with a total of 1859 jar lids, on average 11 per image. Categories include intact (962) versus damaged jar lids (897). 
Types of damages are lid deformations, holes and scratches. An annotation file (jarlids_annots.csv) with bounding boxes is also provided with the dataset.
I use this annotation file to:

1: Crop each image into multiple images where each image contains a single jar lid (cf. figure below).

2: Regroup these cropped images into 2 categories 'intact' and 'damaged'.

This is necessary because Anomalib accepts data only in this format.

![data_rearranged](https://github.com/user-attachments/assets/c9c4e11e-6d98-40c1-9bbd-55825a0b5f7a)

## Setup 
1. Create a conda environment and install Anomalib.
```bash
conda create --name my_anomalib_env python=3.10
conda activate my_anomalib_env
pip install anomalib
anomalib install -v
```
Anomalib could also be installed from its original github repo for more flexibilty, further details on its installation are found in https://github.com/openvinotoolkit/anomalib.

Note: at the time of development of this project, Anomalib had the version v1.1.1.

If you have a GPU and want to use it for the training, make sure that it's correctly connected.

```python
import torch
print(torch.cuda.is_available) # you should get True, if you get False then reinstall a compatible torch version.
```

2. Download the dataset from Kaggle https://www.kaggle.com/datasets/rrighart/jarlids and extract it under the folder cans_defect_detection_dataset_original.

Optionally, you can use the script *visualize_data.py* to view the bounding boxes on the images.
```bash
python visualize_data.py
```

3. Use the script *rearrange_data.py* to prepare the data according to the process described in data preparation section.
```bash
python rearrange_data.py
```
Now you should have intact jar lid images under *cans_defect_detection/intact* and those with defects under *cans_defect_detection/damaged*.

## Training and testing
You can train an Anomalib model on this dataset using:
```bash
python main.py --mode train
```
By default, you'll train the data on 'Padim' model, however you can train on other models such as Patchcore, ReverseDistillation... Check the official documentation of Anomalib for the supported models (https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image). 

The training time might vary depending on your hardware, the model chosen, the number of epochs (and of course the size of the data). For some models, you can interrupt the training after some epochs with ctrl+C. 

What happens in the training is that Anomalib will automatically split the data into training (80% of intact data), validation and test sets, then perform training on the training set.

After training, and before you test, you should get a checkpoint file located under `results/<model name>/cans_defect_detection/<v_some_number>/weights/lightning/model.ckpt`, copy that path and assign it to the variable ckpt_path in the code. Once that is done you can test the model on the test set:
```bash
python main.py --mode test
```
This will output the evaluation metrics such as AUROC and F1 Score, and it will also generate a confusion matrix, which will be saved in the same directory as the Python files. Additionally, you can find the anomaly detection results on the test set under the path `results/<model_name>/cans_defect_detection/<v_some_number>/images/cans_defect_detection_dataset/`.

Some examples of the results:

![confusion_matrix](https://github.com/user-attachments/assets/2297c925-be43-4c5b-b7c4-5ced3eb47534)

![p19_bbox_166_1](https://github.com/user-attachments/assets/57dbe0b6-a9ab-4787-bc18-b3d871936e32)
