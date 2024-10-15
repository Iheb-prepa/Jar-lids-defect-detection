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
