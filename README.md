# Optical Flow estimation from Event Cameras and Spiknig Neural Networks

## Introduction

We present here the code we have used to develop our Spiking Neural Network (SNN), capable of accurately estimating optical flow from event data. Our main contributions are:

- A novel angular loss term which, in conjunction with a standard MSE error function, greatly improves the network's performance and generalization. To the best of our knowledge, we are the first to use such a loss function.
- A model capable of handling temporal dependencies thanks to 3d convolutions on the encoder over consecutive time frames along a temporal axis.
- A stateless SNN, i.e. a model where a reset is performed after each forward pass), that could be implemented on neuromorphic hardware, thus taking advantage of the energy efficiency of such devices.
- accurate results on DSEC (WIP)

We have used the DSEC Dataset (https://dsec.ifi.uzh.ch) to train and evaluate our models, and we report our performance on their official optical flow benchmark.

Our network has been developped with PyTorch (https://pytorch.org/), and we have used the SpikingJelly library (https://github.com/fangwei123456/spikingjelly) for our spiking neuron model, using surrogate gradient learning in our supervised training algorithm. We have used the SpikingJelly version 0.0.0.0.13.

Below can be found an example of the results obtained with our model, evaluated on our validation set. Each quadrant represents:
- Upper-left: Optical Flow ground-truth.
- Upper-right: Masked prediction (only valid pixels are shown)
- Lower-left: optical flow encoding. The estimation at each time step is represented as a Lab image, where the a and b channels represent the x- and y- components of the optical flow, and the L channel accounts for the optical flow magnitude.
- Lower-right: Unmasked prediction. It can be seen that, even if the network was not explicitly trained on the task, it is nonetheless able to obtain a general comprehension of the visual scene, and to show contours (e.g. trafic signals)

![ezgif-2-ec655c32b9](https://user-images.githubusercontent.com/71754039/217257601-344dc0f9-f58c-4981-b9b1-21b46b35751e.gif)

## Installation

We suggest creating a Python virtual environment before running our codes. To do so, you can simply run:
```
python3 -m venv OF_EV_SNN
source OF_EV_SNN/bin/activate
```

Once created, it is only needed to install the packages in the file ```requirements.txt```:
```
pip3 install -r requirements.txt
```

## Codes

Three "core" codes are presented in this repository:
- ```train_3dNet_cat_1res.py```: main code of the repository, containing the training loop and a test loop on both the train and the validation splits.
- ```test_network_metrics.py```: code used to report the metrics of a saved model. Both angular loss and mod loss are reported.
- ```generate_model_prediction.py```: visualization code, it creates a video with the network predictions (see .gif above for an extract of one such video).

We also provide an example network in the ```examples/``` directory, which is our top-performing model on our validation split after 35 epochs of training.

## Data Storage Structure

We suggest placing DSEC in the ```data/``` folder, following the next structure

```
data/
    ├── dataset/
    │      ├─── raw_files/
    │      └─── saved_flow_data/
    │           ├── event_tensors/
    │           │   ├── 11frames/
    │           │   |   ├── left/
    │           │   |   └── ...
    |           |   └── ...
    │           ├── gt_tensors/
    │           ├── mask_tensors/
    │           └── sequence_lists/
    └── ...  
```

The ```raw_files/``` folder contains the data downloaded from the DSEC webpage. After executing the code, different arrays will be created for each chunk, each stored in the corresponding folder. The number of frames per label used can be found in the subdirectories of the ```event_tensors/``` folder (in this case, 11 frames per timestamp = 9ms histograms).
