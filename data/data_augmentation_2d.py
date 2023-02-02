import torch
import torchvision.transforms as TvT

import random

import numpy as np


class Random_horizontal_flip:

    def __init__(self, p = 0.5):

        self.p = p

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:

            events = TvT.functional.hflip(events)

            flow = TvT.functional.hflip(flow)
            flow[:,0] *= -1

            mask = TvT.functional.hflip(mask)

        return (events, flow, mask)

class Random_vertical_flip:

    def __init__(self, p = 0.5):

        self.p = p

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:

            events = TvT.functional.vflip(events)

            flow = TvT.functional.vflip(flow)
            flow[:,1] *= -1

            mask = TvT.functional.vflip(mask)

        return (events, flow, mask)

class Random_event_drop:

    def __init__(self, p = 0.5, min_drop_rate = 0., max_drop_rate = 0.6):

        self.p = p
        self.min_drop_rate = min_drop_rate
        self.max_drop_rate = max_drop_rate

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:

            # probability of an input event to be dropped: random variable uniformly distributed on [TBD, TBD] by default
            q = (self.min_drop_rate - self.max_drop_rate) * torch.rand(1) + self.max_drop_rate

            ev_mask = torch.rand_like(events)
            events = events * (ev_mask > q)

        return (events, flow, mask)

class Random_rotate:

    def __init__(self, p = 0.5):

        self.p = p

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:


            if torch.rand(1).item() <= 0.5:

                events = TvT.functional.rotate(events, -90)

                flow = TvT.functional.rotate(flow, -90)
                flow = np.flip(flow.numpy(),1).copy()   #Reverse of copy of numpy array of given tensor
                flow = torch.from_numpy(flow)
                flow[:,1] *= -1

                mask = TvT.functional.rotate(mask, -90)

            else:

                events = TvT.functional.rotate(events, +90)

                flow = TvT.functional.rotate(flow, +90)
                flow = np.flip(flow.numpy(),1).copy()   #Reverse of copy of numpy array of given tensor
                flow = torch.from_numpy(flow)
                flow[:,0] *= -1

                mask = TvT.functional.rotate(mask, +90)

        return (events, flow, mask)

class Random_patch:

    def __init__(self, p = 0.65, min_size: int = 20, max_size: int = 50, max_patches: int = 6):

        self.p = p

        self.sizes = [i for i in range(min_size, max_size + 1)]

        self.patches = [i for i in range(1, max_patches + 1)]

    def __call__(self, x):

        events, flow, mask = x

        B, C, T, H, W = events.shape

        
        for t in range(T):
            
            if torch.rand(1).item() <= self.p:

                patches = random.choice(self.patches)

                for p in range(patches):

                    size = random.choice(self.sizes)

                    y = random.choice([i for i in range(H - size)])
                    x = random.choice([j for j in range(W - size)])

                    events[:, :, t, y:y+size, x:x+size] = 0


        return (events, flow, mask)


if __name__ == "__main__":

    chunk = torch.randn(4, 2, 256, 256)
    label = torch.randn(4, 2, 256, 256)
    mask = torch.randn_like(label)

    data_augmentation = TvT.Compose([
        Random_event_drop(),
        Random_patch(),
        Random_horizontal_flip(),
        Random_vertical_flip(),
        Random_rotate(),
    ])

    chunk = data_augmentation((chunk, label, mask))
