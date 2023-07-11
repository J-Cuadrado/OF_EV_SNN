import os
import torch
from torch.utils.data import Dataset
import numpy as np

from .indexes import *

class MVSECDataset(Dataset):
    def __init__(self, root:str, sequence: int, transform=None):
        
        
        assert sequence in [1, 2]
        self.sequence = sequence

        self.root = root
        
        if sequence == 1:
            self.length = 5134
        else:
            self.length = 12197

        self.transform = transform        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        data = np.load(os.path.join(self.root, 'outdoor_day{}_{}.npz'.format(self.sequence, str(idx).zfill(6))))
        
        events = data['events']
        mask = ~data['mask']
        label = data['label']

        if self.transform == "events":
            events[:, :, 160:, 35:311] = 0
        elif self.transform == "gt":
            mask[160:, 35:311] = False
            label[:, 160:, 35:311] = 0

        return events, mask, label
        
