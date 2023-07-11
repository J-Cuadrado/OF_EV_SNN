import random

import torch

from spikingjelly.clock_driven import functional

from poolingNet_cat_1res_mvsec import NeuronPool_Separable_Pool3d

from tqdm import tqdm

from mvsec_dataset_v2.mvsec_dataset_outdoor import MVSECDataset 

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

split = 3 

# Create validation dataset
print("Creating Validation Dataset ...")
valid_dataset = MVSECDataset(root = 'files/saved', sequence = 1, transform = None)


# Define validation dataloader
batch_size = 1
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, drop_last = False, pin_memory = True)


########################
## TRAINING FRAMEWORK ##
########################

# Create the network

net = NeuronPool_Separable_Pool3d(multiply_factor = 35.).to(device)
net.load_state_dict(torch.load('results_mvsec/split{}_pretrained/test_epoch22.pth'.format(split)))
net.load_state_dict(torch.load('results_mvsec/maskAll/test_epoch32.pth'.format(split)))

mod_fcn = mod_loss_function
lambda_mod = 1.

ang_fcn = angular_loss_function
lambda_ang = 1.

##########################
## TRAIN, EVAL AND TEST ##
##########################

n_chunks_valid = len(valid_dataloader) # MODIFY/CORRECT THIS PART

#pred_sequence = []
#label_sequence = []

net.eval()

epoch_mod_loss_test = 0.
epoch_ang_loss_test = 0.

acts = []

print('Validating... (test sequence)')

first = True
invalid_labels = 0
for chunk, mask, label in tqdm(valid_dataloader):
    if torch.sum(mask) == 0:
        print('no valid pixels. continuing')
        invalid_labels += 1
        continue
    
    functional.reset_net(net)

    ## The lines below are only needed for outside scenarios
    ## Please comment these lines for indoor flying splits
    chunk[:, :, :, 160:, 35:311] = 0
    mask[:, 160:, 35:311] = False
    label[:, :, 160:, 35:311] = 0.


    mask = torch.unsqueeze(mask, dim = 1)
        
    chunk = chunk.to(device = device, dtype = torch.float32)
    label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
    mask = mask.to(device = device)

    with torch.no_grad():
        _, _, _, pred = net(chunk)
        #pred_list, activations_list = net(chunk)

    #pred = pred_list[-1]
    
    mod_loss = mod_fcn(pred, label, mask)
    ang_loss = ang_fcn(pred, label, mask)

    epoch_mod_loss_test += mod_loss.item() * batch_size
    epoch_ang_loss_test += ang_loss.item() * batch_size
    
    """    
    if first:
        first = False
        for act in activations_list:
            acts.append(torch.mean(act))
    else:
        counter = 0
        for act in activations_list:
            acts[counter] += torch.mean(act)
            counter += 1
    """ 
    

    #pred_sequence.append(torch.squeeze(pred[0,:,:,:]).cpu().detach().numpy())
    #label_sequence.append(torch.squeeze(label[0,:,:,:]).cpu().detach().numpy())
    

epoch_mod_loss_test /= (n_chunks_valid - invalid_labels)
epoch_ang_loss_test /= (n_chunks_valid - invalid_labels)

epoch_loss_valid = epoch_mod_loss_test + epoch_ang_loss_test
print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
    
print({
    'TOT_mod_loss': epoch_mod_loss_test,
    'TOT_ang_loss': epoch_ang_loss_test * 180 / math.pi,
    'TOT_total_loss': epoch_loss_valid,
})

print("Activations")
for a in acts:
    print(a/n_chunks_valid*100)

print('SO FAR, EVERYTHING IS WORKING!!!')
