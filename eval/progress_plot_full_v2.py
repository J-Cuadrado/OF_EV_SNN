import numpy as np
import cv2

from data.utils import vx_vy2v_theta

def plot_evolution(gt, pred, mask, fps, filename = 'comparison.mp4', scale_frame = False):

    n_frames = gt.shape[0]
    H = gt.shape[2]; W = gt.shape[3]

    ## Ground-truth frame generation

    gt_Lab = np.zeros((n_frames, H, W, 3))
    gt_vsincos = vx_vy2v_theta(gt)
    #vmax = np.max(gt_vsincos[:,0,:,:])
    vmax = 256. # A PRIORI FACTOR
    mask_gt = gt_vsincos[:,0,:,:] > vmax

    gt_Lab[:,:,:,0] = 100 * gt_vsincos[:,0,:,:] / vmax
    gt_Lab[:,:,:,1] = 127 * gt_vsincos[:,2,:,:]
    gt_Lab[:,:,:,2] = 127 * gt_vsincos[:,1,:,:]

    gt_Lab[mask_gt] = [[100, 0, 0]]


    ## Pred frame generation

    pred_Lab = np.zeros((n_frames, H, W, 3))
    pred_vsincos = vx_vy2v_theta(pred)
    mask_pred = pred_vsincos[:,0,:,:] > vmax

    pred_Lab[:,:,:,0] = 100 * pred_vsincos[:,0,:,:] / vmax
    pred_Lab[:,:,:,1] = 127 * pred_vsincos[:,2,:,:]
    pred_Lab[:,:,:,2] = 127 * pred_vsincos[:,1,:,:]

    pred_Lab[mask_pred] = [[100, 0, 0]] # White pixels -> saturate pixels with greater value than the maximum gt

    ## Error frame generation

    eps_x = gt[:, 0, :, :] - pred[:, 0, :, :]
    eps_y = gt[:, 1, :, :] - pred[:, 1, :, :]
    eps = np.sqrt(eps_x**2 + eps_y**2)
    eps_max = np.max(eps)
    #print(eps_max)
    #raise

    eps_frames = (255 * eps/eps_max).astype(np.uint8)
    
    ## Scale generation

    L = np.ones((480, 480)) * 100

    a = np.ones((480, 480))
    b = np.ones((480, 480))

    min_value = -127

    for elem in range(480):
        a[:, elem] *= (min_value + elem*(127*2)/479)
        b[elem, :] *= (min_value + elem*(127*2)/479)

    scale_Lab = np.zeros((480, 480, 3))
    scale_Lab[:, :, 0] = L * np.sqrt((a/127)**2 + (b/127)**2)
    scale_Lab[:, :, 1] = a
    scale_Lab[:, :, 2] = b

    outer_circle = np.sqrt(a**2 + b**2) > 127

    #scale_Lab[outer_circle] *= 0 # Black exterior
    scale_Lab[outer_circle] = [[100, 0, 0]]
    
    
    scale_BGR = cv2.cvtColor(scale_Lab.astype(np.float32), cv2.COLOR_LAB2BGR)

    scale_frame = np.ones((H, W, 3))
    scale_frame[:, 80:560, :] = scale_BGR
    scale_frame = (scale_frame * 255).astype(np.uint8)

    ## Video generation
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (3 * W, 2 * H), isColor = True)
    #out = cv2.VideoWriter(filename, fourcc, 1, (2 * W, 2 * H), isColor = True)

    #### NEW PART: MASK PREDICTIONS
    useless_pixels = gt_Lab[:,:,:,0] == 0. # GT greater than 2% of max value
    
 
    masked_pred_Lab = np.copy(pred_Lab)
    masked_pred_Lab[useless_pixels] = [[0, 0, 0]]
    
    eps_frames_masked = np.copy(eps_frames)

    eps_frames_masked[useless_pixels] = 0
    #### NEW PART: MASk PREDICTIONS

    for f in range(n_frames):
        
        gt_BGR = (cv2.cvtColor(gt_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
        pred_BGR = (cv2.cvtColor(pred_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
        masked_pred_BGR = (cv2.cvtColor(masked_pred_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)

        eps_grayscale = (cv2.cvtColor(eps_frames[f], cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
        eps_colormap = (cv2.applyColorMap(eps_grayscale, cv2.COLORMAP_JET)*255).astype(np.uint8)
        
        eps_grayscale_masked = (cv2.cvtColor(eps_frames_masked[f], cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
        eps_colormap_masked = (cv2.applyColorMap(eps_grayscale_masked, cv2.COLORMAP_JET)*255).astype(np.uint8)
        
        frame_gt_pred = np.concatenate((gt_BGR, masked_pred_BGR, pred_BGR), axis = 1)
        frame_eps_scale = np.concatenate((scale_frame, eps_colormap_masked, eps_colormap), axis = 1)

        frame = np.concatenate((frame_gt_pred, frame_eps_scale), axis = 0)

        out.write(frame)
 
    out.release()
 
