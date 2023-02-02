import numpy as np
from tqdm import tqdm
import cv2

def Events2Frames(Events, GT, GT_ts, num_frames_per_ts):

    H = GT.shape[2]
    W = GT.shape[3]

    Events_frames = np.zeros((GT_ts.shape[0]*num_frames_per_ts,2,H,W))
    prev_ts = Events[:,2].min()
  
    print('Cumulating spikes into frames...')
    for t in tqdm(range(GT_ts.shape[0])):
        next_ts = GT_ts[t]
        dt = (next_ts - prev_ts) / num_frames_per_ts
        for n in range(num_frames_per_ts):
            curr_ts = prev_ts + dt
            Events_window = Events[(Events[:,2] >= prev_ts) & (Events[:,2] < curr_ts)]
            for Event in Events_window:
                polarity = int(Event[3]) ## polarity of the event
                coord_x = int(Event[0]) ## x-coordinate (column idx) of the event
                coord_y = int(Event[1]) ## y-coordinate (row idx) of the event
                if polarity == +1:
                    Events_frames[t*num_frames_per_ts + n, 0, coord_y, coord_x] += 1
                else:
                    Events_frames[t*num_frames_per_ts + n, 1, coord_y, coord_x] += 1
            prev_ts = curr_ts

    return Events_frames


def AugmentData(Data, Labels):

    n_frames = Data.shape[0]
    n_labels = Labels.shape[0]

    Data_switched = np.zeros_like(Data)
    Data_mirror = np.zeros_like(Data)

    Data_switched[:, 0, :, :] = Data[:, 1, :, :]
    Data_switched[:, 1, :, :] = Data[:, 0, :, :]

    for i in range(n_frames):
        Data_mirror[i, :, :, :] = Data_switched[n_frames - (i+1), :, :, :]


    Labels_switched = Labels * (-1)
    Labels_mirror = np.zeros_like(Labels)

    for i in range(n_labels):
        Labels_mirror[i, :, :, :] = Labels_switched[n_labels - (i+1), :, :, :]


    TotalData = np.concatenate((Data_mirror, Data), axis = 0).astype(np.float32)
    TotalLabels = np.concatenate((Labels_mirror, Labels), axis = 0).astype(np.float32)
    
    #VideoPlotter(TotalData, TotalLabels, 3)

    return Data_mirror, Labels_mirror


def CreateTrainingSequence(Data, Labels, k1, k2):

    num_batches = k2 // k1
    
    n_frames = Data.shape[0]

    Cin = Data.shape[1]
    Hin = Data.shape[2]
    Win = Data.shape[3]

    Clabel = Labels.shape[1]
    Hlabel = Labels.shape[2]
    Wlabel = Labels.shape[3]

    chunk_sequence = []
    label_sequence = []

    first_label = num_batches - 1
    
    print('Creating training sequence...')

    for i in tqdm(range(0, n_frames - (num_batches - 1) * k2, k2)):
        
        chunk = np.zeros([num_batches, k2, Cin, Hin, Win])
        label = np.zeros([num_batches, Clabel, Hlabel, Wlabel])
        

        for b in range(num_batches):
            chunk[b,:,:,:,:] = Data[i+b*k1:i+b*k1+k2,:,:,:]
            label[b,:,:,:] = Labels[first_label + b, :, :, :]
        
            #print('{}, {}, {}'.format(i+b*k1, i+b*k1+k2, first_label+b))        
        
        chunk_sequence.append(chunk)
        label_sequence.append(label)

        first_label += num_batches
    
    print('Done! \n')
    return chunk_sequence, label_sequence


def VideoPlotter(Data, Labels, num_frames_per_ts):

    n_frames = Labels.shape[0]

    Data_concat = np.concatenate((Data[:,0,:,:], Data[:,1,:,:]), axis = 2)

    for i in range(n_frames * num_frames_per_ts):
        # cv2.imshow('Flow (normalized)',Labels_norm_concat[i,:,:])
        cv2.imshow('Spikes (+ // -)',Data_concat[i,:,:])
        cv2.waitKey(int(50/num_frames_per_ts))
    cv2.destroyAllWindows()

    Labels_norm = cv2.normalize(Labels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Labels_norm_concat = np.concatenate((Labels_norm[:,0,:,:], Labels_norm[:,1,:,:]), axis = 2)

    for i in range(n_frames):
        frame = cv2.applyColorMap(Labels_norm_concat[i,:,:], cv2.COLORMAP_HOT)
        cv2.imshow('Flow (normalized)', frame)
        cv2.waitKey(50)
    cv2.destroyAllWindows()

def vx_vy2v_theta(flow_xy):
    n_labels = flow_xy.shape[0]
    H = flow_xy.shape[2]; W = flow_xy.shape[3]
    flow_vtheta = np.zeros((n_labels, 3, H, W))
    flow_vtheta[:,0,:,:] = np.sqrt(flow_xy[:,0,:,:]**2 + flow_xy[:,1,:,:]**2)
    theta = np.arctan2(flow_xy[:,1,:,:], flow_xy[:,0,:,:])
    flow_vtheta[:,1,:,:] = np.sin(theta)
    flow_vtheta[:,2,:,:] = np.cos(theta)

    thr = 1e-5
    flow_vtheta[:,2][flow_vtheta[:,0] < thr] = 0

    return flow_vtheta


# def CreateTrainingSequence(Data, Labels, k1, k2, num_frames_per_ts):

#     Cin = Data.shape[1]
#     Hin = Data.shape[2]
#     Win = Data.shape[3]

#     Clabel = Labels.shape[1]
#     Hlabel = Labels.shape[2]
#     Wlabel = Labels.shape[3]

#     chunk_sequence = []
#     label_sequence = []

#     chunk_sequence.append(Data[max(0,k1-k2):k1,:,:,:].reshape(1, min(k1,k2), Cin, Hin, Win))
#     label_sequence.append(Labels[int(k1/num_frames_per_ts)-1,:,:,:].reshape(1, Clabel, Hlabel, Wlabel))

#     for i in range(2*k1, Data.shape[0]+1, k1):
#         chunk_sequence.append(Data[i-k2:i,:,:,:].reshape(1, k2, Cin, Hin, Win))
#         label_sequence.append(Labels[int(i/num_frames_per_ts)-1,:,:,:].reshape(1, Clabel, Hlabel, Wlabel))

#     return chunk_sequence, label_sequence
