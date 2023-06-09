import random

import torch

from spikingjelly.clock_driven import functional

from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d

from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite 

from eval.vector_loss_functions import * 

import math

# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

################################
## DATASET LOADING/GENERATION ##
################################

# Define desired temporal resolution (50ms between consecutive layers)
num_frames_per_ts = 11
forward_labels = 1

# Create validation dataset
print("Creating Validation Dataset ...")
valid_dataset = DSECDatasetLite(root = '/data/dataset/saved_flow_data', file_list = 'valid_split_doubleseq.csv', num_frames_per_ts = num_frames_per_ts, stereo = False, transform = None)

# Define validation dataloader
batch_size = 1
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, drop_last = False, pin_memory = True)


########################
## TRAINING FRAMEWORK ##
########################

# Create the network

net = NeuronPool_Separable_Pool3d(multiply_factor = 35.).to(device)
net.load_state_dict(torch.load('examples/checkpoint_epoch34.pth'))

mod_fcn = mod_loss_function
lambda_mod = 1.

ang_fcn = angular_loss_function
lambda_ang = 1.

###############################
## COMPUTE MODEL PERFORMANCE ##
###############################

n_chunks_valid = len(valid_dataloader)

net.eval()

epoch_mod_loss_test = 0.
epoch_ang_loss_test = 0.

print('Validating... (test sequence)')

for chunk, mask, label in tqdm(valid_dataloader):

    functional.reset_net(net)

    chunk = torch.transpose(chunk, 1, 2)

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
    

epoch_mod_loss_test /= n_chunks_valid
epoch_ang_loss_test /= n_chunks_valid

epoch_loss_valid = epoch_mod_loss_test + epoch_ang_loss_test
print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
    
print({
    'TOT_mod_loss': epoch_mod_loss_test,
    'TOT_ang_loss': epoch_ang_loss_test * 180 / math.pi,
    'TOT_total_loss': epoch_loss_valid,
})

print('SO FAR, EVERYTHING IS WORKING!!!')
