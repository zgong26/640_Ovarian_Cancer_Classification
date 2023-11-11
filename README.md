# 640_Ovarian_Cancer_Classification
For CS640, Fall 2023 project at BU

## Notes
Make sure you store all original images to 'train_images_compressed_80' to run transformer.py
### main.py
Load tensors from 'modified_images_tensors' foler to model and starts training process. Whenever an epoch that yields a better accuracy, it saves the current .pt PyTorch model file.
### continue_training.py
Continue the training process by loading .pt file saved previously.
### transformer.py
Transfrom original images from 'train_images_compressed_80' folder to tensors and save them to 'modified_images_tensors' folder
### GPU_Usage.py
Prints out current GPU usage to check if GPU is used rather than CPU
### Playground.py
Just scratch paper, should not affect the project's functionality
