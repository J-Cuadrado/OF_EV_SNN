import torch
import numpy as np

def mod_loss_function(pred, label, mask):

    n_pixels = torch.sum(mask)
    #print(torch.sum(pred), torch.sum(label))    
    #print(pred)
    error_mod = torch.sqrt(torch.pow(pred[:,0] - label[:,0], 2) + torch.pow(pred[:,1] - label[:,1], 2))
    
    return torch.sum(error_mod * mask) / n_pixels
    
def rel_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
    #print(torch.sum(pred), torch.sum(label))    
    #print(pred)
    error_mod = torch.sqrt(torch.pow(pred[:,0] - label[:,0], 2) + torch.pow(pred[:,1] - label[:,1], 2))
    gt_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))
    
    return (1/n_pixels) * torch.sum((error_mod * mask) / (gt_mod + epsilon))


def cosine_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
   
    pred_mod = torch.sqrt(torch.pow(pred[:,0], 2) + torch.pow(pred[:,1], 2))
    label_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))

    dot_product = pred[:,0]*label[:,0] + pred[:,1]*label[:,1]

    cosine = (dot_product + epsilon) / (pred_mod*label_mod + epsilon)
    #cosine = (dot_product) / (pred_mod*label_mod)
    
    cosine = torch.clamp(cosine, min = -1. + epsilon, max = 1. - epsilon)

    return torch.sum((1. - cosine) * mask) /  n_pixels


def angular_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
   
    pred_mod = torch.sqrt(torch.pow(pred[:,0], 2) + torch.pow(pred[:,1], 2))
    label_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))

    dot_product = pred[:,0]*label[:,0] + pred[:,1]*label[:,1]

    cosine = (dot_product + epsilon) / (pred_mod*label_mod + epsilon)
    #cosine = (dot_product) / (pred_mod*label_mod)
    
    cosine = torch.clamp(cosine, min = -1. + epsilon, max = 1. - epsilon)

    return torch.sum(torch.acos(cosine) * mask) /  n_pixels
