# 640_Ovarian_Cancer_Classification
For CS640, Fall 2023 project at BU

## Overview
### Dataset Expansion through Image Slicing: 
Each image is segmented into multiple squares to enrich the dataset and enhance feature detection.
### Relabeling of Segmented Squares: 
Each segmented square is relabeled, preparing the dataset for effective supervised learning. The relabeled csv file is called averaged_train.csv, it is also modified in such a way that each label contains same amount of data/square images to avoid bias.
### Training with ResNet50 Model: 
Utilizes the ResNet50 architecture to train on the newly created and labeled image tensors, leveraging its advanced image recognition capabilities. 1/5 of dataset is used as validation set.
### Classification and Prediction: 
Employs the trained ResNet50 model for accurate classification and prediction of new image data.

## Notes
Make sure you store all original images to 'train_images_compressed_80' to SquareSlicing.py. A trained model is provided attached in the repo.
### start_training.py
Load tensors from 'modified_images_tensors' folder to model and starts training process. It saves each .pt PyTorch model file after each epoch.
Note: 100% GPU Consumption
### continue_training.py
Continue the training process by loading .pt file saved previously.
Note: 100% GPU Consumption
### SquareSlicing.py
Segment images into content-rich squares, either saving them as tensors or JPEGs for deep learning, and efficiently processes these using a multi-threaded approach. It concludes by organizing the processed data into a CSV file, mapping image IDs to their corresponding labels.
Note: It uses 16 threads in CPU by default. Adjust max_worker in process_dataset_multithreaded function accordingly.
### testing.py
Let model to predict labels in test_images_compressed_80 folder.
