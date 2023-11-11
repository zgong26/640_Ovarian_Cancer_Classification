# 640_Ovarian_Cancer_Classification
For CS640, Fall 2023 project at BU

## Notes
Make sure you store all original images to 'train_images_compressed_80' to run transformer.py
### main.py
Load tensors from 'modified_images_tensors' foler to model and starts training process. Whenever an epoch that yields a better accuracy, it saves the current .pt PyTorch model file.
Note: 100% GPU Consumption
### continue_training.py
Continue the training process by loading .pt file saved previously.
Note: 100% GPU Consumption
### transform_image.py
Transfrom original images from 'train_images_compressed_80' folder to tensors and save them to 'modified_images_tensors' folder.
Implemented a SmartCenterCrop class to automatically and randomly choose a square that contains the most tissues. The square is then sent to the pipline to be generated to tensors.
Note: Use 16 threads in CPU by default. Adjust max_worker in process_dataset_multithreaded function accordingly.
### GPU_Usage.py
Prints out current GPU usage to check if GPU is used rather than CPU
### Playground.py
Just scratch paper, should not affect the project's functionality
