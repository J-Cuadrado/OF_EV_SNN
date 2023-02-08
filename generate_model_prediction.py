import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as TvT

from torch.autograd import Variable

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import neuron

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d


from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite

import numpy as np

from eval.progress_plot_full_v2 import plot_evolution

import math
import os


# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


################################
## DATASET LOADING/GENERATION ##
################################

num_frames_per_ts = 1
forward_labels = 1

# Create validation dataset
print("Creating Validation Dataset ...")
valid_dataset = DSECDatasetLite(root = '/opt/Partage/DSEC/saved_flow_data', file_list = 'valid_split_doubleseq.csv', num_frames_per_ts = 11, transform = None)

# Define validation dataloader
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, drop_last = False, pin_memory = True)

########################
## TRAINING FRAMEWORK ##
########################

# Create the network

net = NeuronPool_Separable_Pool3d().to(device)
net.load_state_dict(torch.load('examples/checkpoint_epoch34.pth'))


##########
## TEST ##
##########

# Validation Dataset


pred_sequence = []
label_sequence = []
mask_sequence = []

 
net.eval()
print('Validating... (test sequence)')

net.eval()
for chunk, mask, label in tqdm(valid_dataloader):

    functional.reset_net(net)
    chunk = torch.transpose(chunk, 1, 2)

    
    mask = torch.unsqueeze(mask, dim = 1)
    mask = torch.cat((mask, mask), axis = 1)
        
    chunk = chunk.to(device = device, dtype = torch.float32)

    label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]

    mask = mask.to(device = device)

    with torch.no_grad():
        _, _, _, pred = net(chunk)

    pred_sequence.append(torch.squeeze(pred[0,:,:,:]).cpu().detach().numpy())
    label_sequence.append(torch.squeeze(label[0,:,:,:]).cpu().detach().numpy())
    mask_sequence.append(torch.squeeze(mask[0]).cpu().detach().numpy())

    
# Video generation
pred_sequence = np.array(pred_sequence)
label_sequence = np.array(label_sequence)

videofile = 'results/flow_valid_epoch34.mp4'

plot_evolution(label_sequence, pred_sequence, mask_sequence, 10, videofile)

print('SO FAR, EVERYTHING IS WORKING!!!')
