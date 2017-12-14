# Semantic Segmentation Framework in Tensorflow

The following repository contains a functional framework to perform semantic segmentation using Convolutional Neural Networks in Tensorflow.

## Actually in development

### Functionalities implemented:
- Train, validation and test in one run using a proper configuration.
- Early stopping with model saving control via different metrics
- Test as a prediction system only
- Configuration file to run experiments easily
- Metrics: mean accuracy, mean IoU and confusion matrix visible in TensorBoard
- Models: DenseNetFCN (tiramisu)

### How to run it
- Configure the configuration file in config/ConfigFile.py
- Run code using: CUDA_VISIBLE_DEVICES=[gpu_number] python main.py --exp_name [experiment_name] 
  --exp_folder [path_to_save_experiment] --config_file [path_to_config_file]
  You can define default values to this input arguments in main.py
  
### Actual limitations
- Only accepts one dataset where RGB images must match the GT images in name, quantity and order. Folder paths for each subset like train, validation and test can be defined in the config file with the respective names of the subfolders path for RGB and GT images.
- Data loader using numpy
