import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class DSECDatasetLite(Dataset):
    def __init__(self, root:str, file_list: str, num_frames_per_ts: int = 1, stereo = False, transform=None):
        
        self.events_path = os.path.join(root, 'event_tensors', '{}frames'.format(str(num_frames_per_ts).zfill(2)))
        self.flow_path = os.path.join(root, 'gt_tensors')
        self.mask_path = os.path.join(root, 'mask_tensors')

        self.stereo = stereo

        sequence_file = os.path.join(root, 'sequence_lists', file_list)

        self.files = pd.read_csv(sequence_file, header = None)

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        target_file_1 = self.files.iloc[idx, 0]
        target_file_2 = self.files.iloc[idx, 1]

        eventsL1 = torch.from_numpy(np.load(os.path.join(self.events_path, target_file_1)))
        eventsL2 = torch.from_numpy(np.load(os.path.join(self.events_path, target_file_2)))
        eventsL = torch.cat((eventsL1, eventsL2), axis = 0)
        
        if self.stereo:
            eventsR1 = torch.from_numpy(np.load(os.path.join(self.events_path, 'right', target_file_1)))
            eventsR2 = torch.from_numpy(np.load(os.path.join(self.events_path, 'right', target_file_2)))
            eventsR = torch.cat((eventsR1, eventsR2), axis = 0)

            eventsL = torch.cat((eventsL, eventsR), axis = 1)
        
        mask = torch.from_numpy(np.load(os.path.join(self.mask_path, target_file_2)))
        label = torch.from_numpy(np.load(os.path.join(self.flow_path, target_file_2)))
        
        if self.transform:
            eventsL = self.transform(events)
            mask = self.transform(mask)
            label = self.transform(label)
        
        return eventsL[-21:], mask, label
