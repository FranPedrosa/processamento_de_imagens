from skimage import color
import numpy as np


def less_colors(src):
    w,h,_ = src.shape
    dst = np.zeros((w,h)).astype(int)

    h = color.rgb2hsv(src)[:,:,0]
    s = color.rgb2hsv(src)[:,:,1]
    v = color.rgb2hsv(src)[:,:,2]

    #mask = s < 0.3
    #vs = v // 0.25 + 10
    #dst[mask] = vs[mask]

    mask = s > 0.3
    hs = ((h+ 1/14) // (1/7) % 7)+ 1
    dst[mask] = hs[mask]

    mask = np.logical_and(s < 0.3,v < 0.25)
    dst[mask] = 8
    mask = np.logical_and(s < 0.3,v > 0.75)
    dst[mask] = 9

    
    return dst

def visualize(src):
    w,h = src.shape
    dst = np.zeros((w,h,3)).astype(int)

    dst[:,:] = [100,100,100]

    dst[src == 1] = [255,0,0]
    dst[src == 2] = [255,255,0]
    dst[src == 3] = [0,255,0]
    dst[src == 4] = [0,255,255]
    dst[src == 5] = [0,0,255]
    dst[src == 6] = [100,0,255]
    dst[src == 7] = [255,0,255]

    dst[src == 8] = [0,0,0]
    dst[src == 9] = [255,255,255]

    return dst
    

def reds(src):
    img = src.copy()
    h = color.rgb2hsv(img)[:,:,0]
    s = color.rgb2hsv(img)[:,:,1]
    h = 2*np.abs(0.5 - h)
    mask_h = h > 0.95
    mask_s = s > 0.30
    mask = np.logical_and(mask_h,mask_s) 
    return mask

def blues(src):
    img = src.copy()
    h = color.rgb2hsv(img)[:,:,0]
    s = color.rgb2hsv(img)[:,:,1]
    h = np.abs(0.66 - h)
    mask_h = h < 0.1
    mask_s = s > 0.5
    mask = np.logical_and(mask_h,mask_s) 
    return mask

def greens(src):
    img = src.copy()
    h = color.rgb2hsv(img)[:,:,0]
    s = color.rgb2hsv(img)[:,:,1]
    h = np.abs(0.33 - h)
    s = np.abs(0.5 - h)
    mask_h = h < 0.1
    mask_s = s < 0.25
    mask = np.logical_and(mask_h,mask_s) 
    return mask

def yellows(src):
    img = src.copy()
    h = color.rgb2hsv(img)[:,:,0]
    s = color.rgb2hsv(img)[:,:,1]
    h = np.abs(0.16 - h)
    mask_h = h < 0.1
    mask_s = np.logical_and(s > 0.45,s < 0.6)
    mask = np.logical_and(mask_h,mask_s) 
    return mask

def oranges(src):
    img = src.copy()
    h = color.rgb2hsv(img)[:,:,0]
    s = color.rgb2hsv(img)[:,:,1]
    h = np.abs(0.10 - h)
    mask_h = h < 0.1
    mask_s = s > 0.6
    mask = np.logical_and(mask_h,mask_s) 
    return mask


def blacks(src):
    img = src.copy()
    v = color.rgb2hsv(img)[:,:,2]
    s = color.rgb2hsv(img)[:,:,1]
    mask_v = v < 0.2
    mask_s = s < 0.2
    mask = np.logical_and(mask_v,mask_s) 
    return mask

def whites(src):
    img = src.copy()
    v = color.rgb2hsv(img)[:,:,2]
    s = color.rgb2hsv(img)[:,:,1]
    mask_v = v > 0.8
    mask_s = s < 0.2
    mask = np.logical_and(mask_v,mask_s) 
    return mask

