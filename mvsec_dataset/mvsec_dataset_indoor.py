import os
import torch
from torch.utils.data import Dataset
import numpy as np

from .indexes import *

class MVSECDataset(Dataset):
    def __init__(self, root:str, split: int, condition: str, transform=None):
        
        assert split in [1, 2, 3]
        assert condition in ['train', 'valid', 'test']
        
        self.root = root
        
        self.files = []
        
        if condition == 'train':
        
            if split == 1:
                for seq in [2, 3]:
                    data_range = SEQUENCES_FRAMES['indoor_flying']['split1']['indoor_flying{}'.format(seq)]
                    for idx in range(data_range[0], data_range[1]):
                        self.files.append('indoor_flying{}_{}.npz'.format(seq, str(idx).zfill(6)))
                        
            elif split == 2:
                for seq in [1, 3]:
                    data_range = SEQUENCES_FRAMES['indoor_flying']['split2']['indoor_flying{}'.format(seq)]
                    for idx in range(data_range[0], data_range[1]):
                        self.files.append('indoor_flying{}_{}.npz'.format(seq, str(idx).zfill(6)))
                        
            elif split == 3:
                for seq in [1, 2]:
                    data_range = SEQUENCES_FRAMES['indoor_flying']['split3']['indoor_flying{}'.format(seq)]
                    for idx in range(data_range[0], data_range[1]):
                        self.files.append('indoor_flying{}_{}.npz'.format(seq, str(idx).zfill(6)))
        
        elif condition == 'valid':
            if split == 1:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split1']['indoor_flying1']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT1_VALID_INDICES:
                    self.files.append('indoor_flying1_{}.npz'.format(str(data_idx[idx]).zfill(6)))
            elif split == 2:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split2']['indoor_flying2']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT2_VALID_INDICES:
                    self.files.append('indoor_flying2_{}.npz'.format(str(data_idx[idx]).zfill(6)))
            elif split == 3:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split3']['indoor_flying3']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT3_VALID_INDICES:
                    self.files.append('indoor_flying3_{}.npz'.format(str(data_idx[idx]).zfill(6)))
        
        elif condition == 'test':
            if split == 1:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split1']['indoor_flying1']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT1_TEST_INDICES:
                    self.files.append('indoor_flying1_{}.npz'.format(str(data_idx[idx]).zfill(6)))
            elif split == 2:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split2']['indoor_flying2']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT2_TEST_INDICES:
                    self.files.append('indoor_flying2_{}.npz'.format(str(data_idx[idx]).zfill(6)))
            elif split == 3:
                data_range = SEQUENCES_FRAMES['indoor_flying']['split3']['indoor_flying3']
                data_idx = [i for i in range(data_range[0], data_range[1])]
                for idx in SPLIT3_TEST_INDICES:
                    self.files.append('indoor_flying3_{}.npz'.format(str(data_idx[idx]).zfill(6)))



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        data = np.load(os.path.join(self.root, self.files[idx]))
        
        return data['events'], ~data['mask'], data['label']
        
