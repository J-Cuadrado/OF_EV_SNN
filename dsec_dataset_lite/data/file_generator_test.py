import os

import imageio

import numpy as np
import pandas as pd

from .event2frame import EventSlicer
from .event2frame import rectify_events, cumulate_spikes_into_frames

import h5py

from tqdm import tqdm


def generate_files(root: str, sequence: str, num_frames_per_ts: int = 1):


    timestamps = pd.read_csv(os.path.join(root, 'test', sequence, '{}.csv'.format(sequence)), delimiter = ',', dtype='int64').to_numpy()

    eventsL_path = os.path.join(root, 'test', sequence, 'events', 'left')

    save_path_events = os.path.join(root, 'saved_flow_data', 'test', 'event_tensors',  '{}frames'.format(str(num_frames_per_ts).zfill(2)), 'left')
    print(save_path_events)
    
    _load_events(sequence, num_frames_per_ts, eventsL_path, timestamps, save_path_events)






def _create_flow_maps(sequence: str, flow_maps_path, save_path_flow, save_path_mask):

    flow_maps_list = os.listdir(flow_maps_path)
    flow_maps_list.sort()

    img_idx = 0

    for flow_map in flow_maps_list:

        img_idx += 1

        path_to_flowfile = os.path.join(flow_maps_path, flow_map)

        flow_16bit = imageio.imread(path_to_flowfile, format='PNG-FI')

        flow_x = (flow_16bit[:,:,0].astype(float) - 2**15) / 128.
        flow_y = (flow_16bit[:,:,1].astype(float) - 2**15) / 128.
        valid_pixels = flow_16bit[:,:,2].astype(bool)


        flow_x = np.expand_dims(flow_x, axis=0)  # shape (H, W) --> (1, H, W)
        flow_y = np.expand_dims(flow_y, axis=0)


        flow_map = np.concatenate((flow_x, flow_y), axis = 0).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(img_idx).zfill(4))

        np.save(os.path.join(save_path_flow, filename), flow_map)
        np.save(os.path.join(save_path_mask, filename), valid_pixels)


def _load_events(sequence, num_frames_per_ts, events_path, timestamps, save_path_events):

    # load data
    datafile_path = os.path.join(events_path, "events.h5")
    datafile = h5py.File(datafile_path, 'r')
    event_slicer = EventSlicer(datafile)

    N_chunks = timestamps.shape[0]  # N_chunks = N_grountruths

    for numchunk in tqdm(range(N_chunks)):

        fileidx = timestamps[numchunk, 2]

        t_end = timestamps[numchunk, 1]
        dt = (100000) / num_frames_per_ts
        t_beg = t_end - 21*dt

        chunk = []

        for numframe in range(21):

            t_start = t_beg + numframe * dt
            t_end = t_beg + (numframe + 1) * dt

            # load events within time window
            event_data = event_slicer.get_events(t_start, t_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            # rectify events
            rectmap_path = os.path.join(events_path, "rectify_map.h5")
            rectmap_file = h5py.File(rectmap_path)
            rectmap = rectmap_file['rectify_map'][()]

            xy_rect = rectify_events(x, y, rectmap)  # TODO change 'location' for rectification map !
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            # cumulate events
            frame = cumulate_spikes_into_frames(x, y, p)
            chunk.append(frame)

        # format into chunks
        chunk = np.array(chunk).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(fileidx).zfill(6))


        np.save(os.path.join(save_path_events, filename), chunk)


    # close hdf5 files
    datafile.close()
    rectmap_file.close()
