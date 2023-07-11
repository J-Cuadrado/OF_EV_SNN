import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as TvT

from torch.autograd import Variable

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import neuron

from network_3d.poolingNet_cat_1res_mvsec import NeuronPool_Separable_Pool3d


from tqdm import tqdm

from mvsec_dataset_v2.mvsec_dataset_outdoor_v2 import MVSECDataset 
from data.data_augmentation_2d import *

import numpy as np

from pytorch_msssim import MS_SSIM

from eval.vector_loss_functions import * 
from eval.metrics_v2 import compute_metrics

from eval.progress_plot_cropped import plot_evolution

import math
import os


# Enable GPU
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def set_random_seed(seed):
    #Python
    random.seed(seed)

    #Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#    if int(torch.__version__.split('.')[1]) < 8: # for pytorch < 1.8
#       torch.set_deterministic(True)
#    else:
#       torch.use_deterministic_algorithms(True)

    # NumPy
    np.random.seed(seed)

seed = 2305
set_random_seed(seed)

################################
## DATASET LOADING/GENERATION ##
################################

# Define desired temporal resolution (50ms between consecutive layers)
num_frames_per_ts = 11
forward_labels = 1

split = 3 

# Create training dataset
print("Loading Training Dataset ...")
#train_dataset = MVSECDataset(root = 'mvsec_dataset_v2/files', split = split, condition = 'train', transform = None)
train_dataset = MVSECDataset(root = '/home/javierca/mvsec_dataset_v2/files_parallel', sequence = 2, transform = "gt")

# Define training dataloader
batch_size = 1
batch_multiplyer = 1
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, pin_memory = True)


# Create validation dataset
print("Loading Validation Dataset ...")
#valid_dataset = MVSECDataset(root = 'mvsec_dataset_v2/files', split = split, condition = 'valid', transform = None)
valid_dataset = MVSECDataset(root = '/opt/Partage/MVSEC/outdoor_day1_files', sequence = 1, transform = "gt")

# Define validation dataloader
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, drop_last = False, pin_memory = True)


########################
## TRAINING FRAMEWORK ##
########################

# Create the network

multiply_factor = 35.
net = NeuronPool_Separable_Pool3d(multiply_factor = multiply_factor).to(device)

trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Trainable parameters: {}'.format(trainable_params))

net.load_state_dict(torch.load('examples/checkpoint_epoch34.pth')) # If using a pre-trained model as a departure point


# Initialize network weights

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        print(m)
        nn.init.xavier_uniform_(m.weight)


# Create the optimizer
lr = 2e-4
wd = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay = wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20, 35], gamma = 0.5)

# Define the loss function

mod_fcn = mod_loss_function
lambda_mod = 1.

ang_fcn = angular_loss_function
lambda_ang = 1.


# Define the number of epochs

n_epochs = 35

###################
## TRAINING LOGS ##
###################

# Decide whether or not to store the network
save_net = True
test_acc = float('inf') 
is_best = False


#####################
## SETUP FUNCTIONS ##
#####################

# Create data augmentation pipeline
data_augmentation = TvT.Compose([
#    Random_event_drop(),
#    Random_patch(p = 0.8),
    Random_horizontal_flip(p = 0.4),
#    Random_vertical_flip(p = 0.15), # change horizontal to 0.3 when using this, 0.4 otherwise
#    Random_rotate(p = 0.1), # change vertical to 0.1 when using this, 0.15 otherwise
])

##########################
## TRAIN, EVAL AND TEST ##
##########################

n_chunks_train = len(train_dataloader)
n_chunks_valid = len(valid_dataloader)

for epoch in range(n_epochs):

    print(f'Epoch {epoch}')

    net.train()
    
    running_loss = 0.

    epoch_mod_loss = 0.
    epoch_ang_loss = 0.

    batch_iter = 0

    print('Training...')
    for chunk, mask, label in tqdm(train_dataloader):

        if torch.sum(mask) == 0:
            print('no valid pixels. continuing')
            continue

        functional.reset_net(net)


        mask = torch.unsqueeze(mask, dim = 1)
        
        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
        chunk = chunk.to(device = device, dtype = torch.float32)
        mask = mask.to(device = device)

        chunk, label, mask = data_augmentation([chunk, label, mask])

        pred_list = net(chunk)
        
        mod_loss = 0.
        ang_loss = 0.
        curr_loss = 0.

        for pred in pred_list:
            mod_loss += mod_fcn(pred, label, mask)
            ang_loss += ang_fcn(pred, label, mask)
        
            curr_loss += lambda_mod * mod_loss + lambda_ang * ang_loss
        

        
        if np.isnan(curr_loss.item()):
            raise

        curr_loss.backward()

        if batch_iter % batch_multiplyer == 0:
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()

            optimizer.zero_grad()

        
        running_loss += curr_loss.item() * batch_size

        epoch_mod_loss += mod_loss.item() * batch_size
        epoch_ang_loss += ang_loss.item() * batch_size



    epoch_loss = running_loss / n_chunks_train

    epoch_mod_loss /= n_chunks_train
    epoch_ang_loss /= n_chunks_train
    
    print(f'Epoch loss = {epoch_loss}')


    # Training Dataset (eval)

    net.eval()

    epoch_mod_loss = 0.
    epoch_ang_loss = 0.

    print('Validating... (training sequence)')

    for chunk, mask, label in tqdm(train_dataloader):

        if torch.sum(mask) == 0:
            print('no valid pixels. continuing')
            continue

        functional.reset_net(net)
        
        mask = torch.unsqueeze(mask, dim = 1)

        chunk = chunk.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
        mask = mask.to(device = device)
        
        with torch.no_grad():
            _, _, _, pred = net(chunk)

        mod_loss = mod_fcn(pred, label, mask)
        ang_loss = ang_fcn(pred, label, mask)

        epoch_mod_loss += mod_loss.item() * batch_size
        epoch_ang_loss += ang_loss.item() * batch_size


    epoch_mod_loss /= n_chunks_train
    epoch_ang_loss /= n_chunks_train

    epoch_loss_train_eval = epoch_mod_loss + epoch_ang_loss
    print('Epoch loss (Validation): {} \n'.format(epoch_loss_train_eval))



    # Validation Dataset

    pred_sequence = []
    label_sequence = []

    net.eval()

    epoch_mod_loss_test = 0.
    epoch_ang_loss_test = 0.

    print('Validating... (test sequence)')

    for chunk, mask, label in tqdm(valid_dataloader):

        if torch.sum(mask) == 0:
            print('no valid pixels. continuing')
            continue
        
        functional.reset_net(net)

        mask = torch.unsqueeze(mask, dim = 1)
        
        chunk = chunk.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
        mask = mask.to(device = device)

        with torch.no_grad():
            _, _, _, pred = net(chunk)

        
        mod_loss = mod_fcn(pred, label, mask)
        ang_loss = ang_fcn(pred, label, mask)

        epoch_mod_loss_test += mod_loss.item() * batch_size
        epoch_ang_loss_test += ang_loss.item() * batch_size
        

        pred_sequence.append(torch.squeeze(pred[0,:,:,:]).cpu().detach().numpy())
        label_sequence.append(torch.squeeze(label[0,:,:,:]).cpu().detach().numpy())
    

    epoch_mod_loss_test /= n_chunks_valid
    epoch_ang_loss_test /= n_chunks_valid

    epoch_loss_valid = epoch_mod_loss_test + epoch_ang_loss_test
    print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
    
    # Save the network

    if save_net & (epoch_loss_valid < test_acc):
        
        is_best = True        

        test_acc = epoch_loss_valid
        torch.save(net.state_dict(), 'results_mvsec/outdoor/test_epoch{}.pth'.format(epoch))
    
    
        is_best = False


    scheduler.step()

print('SO FAR, EVERYTHING IS WORKING!!!')
