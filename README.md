# Satellite Image Segmentation with Convolutional Neural Networks
##### Tariq Kalim, Bayrem Kaabachi, and Ghassen Karray
In this project we tackle the Road Sgmentation challenge on AICrowd using a U-Net architecture and data augmentation as a performance "boost". We managed to achieved a F1 score of 0.907 
on the the test set.

The details of our techniques can be found in the report.pdf file.

This repository includes a pre-trained model that can be used to generate predictions, as well as the code to train the model from scratch. 

### Libraries
The following libraries must be installed to run the project:

- Pytorch 1.3.1
- torchvision 0.4.2

PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
We mostly chose to use pytorch for the ease of use and understandability of the code.


### Setup
The model has been trained using GPU acceleration on google colab, with the following setup:

GPU: 1xTesla K80 , having 2496 CUDA cores, compute 3.7,  12GB(11.439GB Usable) GDDR5  VRAM
 
CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 45MB Cache
 
RAM: ~12.6 GB Available
 
Disk: ~320 GB Available 
 
Due to running on google colab for every 12hrs or so Disk, RAM, VRAM, CPU cache etc data that is on our alloted virtual machine will get erased 

### How to run
To avoid re-training the model, we have provided its weights in the file `model_final_final`. Therefore, to generate the predictions, you need to run the script `run.py`. The test set images must be put in the directory `test_set_images`.
It is possible to use either the CPU or GPU, if we detect that "cuda" is available the program will run on gpu.
the run.py generate prediction in a Predictions folder. then you would need to run `run -i mask_to_submission.py` to generate a csv file from the predictions called "dummy_submission".


### How to train
We have provided a notebook `road-segmentation.ipynb` that can be used to train the model from scratch.
To configure the number of epochs and batch size you should modify the "config.py" file.

### Description of the files
All models are grouped into classes, in order to improve code readability and reusability. The following three models have been supplied:

- `Models/UNet.py`: this model classifies all patches as background, and has been used a baseline and for debug purposes.
- `Models/Resnet50.py`: classifier based on logistic regression.
- `Models/Resnet101`: classifier based on convolutional neural networks. This file contains the neural network structure, as well as the training parameters.

Furthermore, the following files contain utility methods:
- `helpers.py`: contains the image processing methods.
- `train.py`: contains the training of the model depending if cuda is there or not
- `predict.py`: contains the prediction of the model depending if cuda is there or not
- `transformations.py`: contains the methods that let us do transformations on images i.e padding,rotating etc. It acts just like pytorch transforms but it also does the transformation on the groundtruth.
- `validate.py`: this toolkit can be used to get validation score
- `datasets.py`: contains the files that load the dataset


Finally, the following scripts are the ones that can be run:
- `run.py`: reads the weights from a file and performs classification on the test set, using convolutional neural networks.
- `road-segmentation.ipynb`: this notebook can be used to train the network from scratch.


To see gpu activity while doing Deep Learning tasks, use command 'nvidia-smi' it'll show an output like this :

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   31C    P8    27W / 149W |      0MiB / 11441MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+