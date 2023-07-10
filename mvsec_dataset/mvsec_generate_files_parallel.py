import numpy as np

import h5py
import os

from mvsec_fcns import *

from joblib import Parallel, delayed

def get_events_flow_mask(idx):

    # Events
    ts = timestamps[idx]
    beg_time = ts - nframes * dt
    frame_seq = np.zeros((2, nframes, H, W))
    for f in range(nframes):
        event_window = events[(events[:,2] >= beg_time + f*dt) & (events[:,2] < beg_time + (f+1)*dt)]
        for event in event_window:
            polarity = int(event[3]) ## polarity of the event
            coord_x = int(event[0]) ## x-coordinate (column idx) of the event
            coord_y = int(event[1]) ## y-coordinate (row idx) of the event
            if polarity == +1:
                frame_seq[0, f, coord_y, coord_x] += 1
            else:
                frame_seq[1, f, coord_y, coord_x] += 1

    # GT
    flow = np.zeros((2, H, W))
    flow[0] = flow_x[idx]; flow[1] = flow_y[idx]
        
    # Mask
    thr = 1e-5
    mask = (flow_x[idx] <= thr) & (flow_y[idx] <= thr)

    # Create savefile
    #savefile = os.path.join(dataset_path, 'saved_data', '{}_{}.npz'.format(sequence, str(idx).zfill(6)))
    savefile = os.path.join('files/saved', '{}_{}.npz'.format(sequence, str(idx).zfill(6)))
    np.savez(savefile, events = frame_seq, label = flow, mask = mask)



dataset_path = 'files/raw'
sequences = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'outdoor_day2']

# Image definition (constant throughout the dataset)
H = 260; W = 346


# MVSEC WORKS AT 20Hz, whereas DSEC works at 10Hz
# As such, not the same number of frames per label must be specified

dt = 100/11 * 1e-3
nframes = 21

for sequence in sequences:

    # Loading event data
    
    data_file = h5py.File(os.path.join(dataset_path, '{}_data.hdf5'.format(sequence)), 'r')
    events = np.array(data_file['davis']['left']['events'])
    data_file.close()
 
    # Event Rectification

    rect_map_x = np.loadtxt(os.path.join(dataset_path, 'outdoor_day_left_x_map.txt'))
    rect_map_y = np.loadtxt(os.path.join(dataset_path, 'outdoor_day_left_y_map.txt'))

    events = mvsecRectifyEvents(events, rect_map_x, rect_map_y)
 
    # Loading gt data

    gt_file = np.load(os.path.join(dataset_path, '{}_gt_flow_dist.npz'.format(sequence)))

    timestamps = gt_file['timestamps']
    flow_x = gt_file['x_flow_dist']; flow_y = gt_file['y_flow_dist']



    # Generate event, mask and gt tuples

    Parallel(n_jobs=32)(delayed(get_events_flow_mask)(idx) for idx in range(len(timestamps)))
