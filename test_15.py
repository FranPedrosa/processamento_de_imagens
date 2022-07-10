from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from split_colors import less_colors

def neighbors_4(center):
    x,y = center[0],center[1]
    return (x+1,y),(x,y-1),(x-1,y),(x,y+1)


def valid(p,w,h):
    return 0 <= p[0] and p[0] < w and 0 <= p[1] and p[1] < h 

def get_region(mask,seed):
    region = [seed]
    w,h = mask.shape
    pos = 0
    while pos < len(region):
        p = region[pos]
        for n in neighbors_4(p):
            if valid(n,w,h) and mask[n] and not n in region:
                region.append(n)
        pos += 1
    return np.array(region).T

def split_regions(src):
    mask = src.copy()
    regions = np.zeros(src.shape,dtype=np.uint8)
    i = 1
    while np.any(mask):
        seed = tuple(np.argwhere(mask)[0])
        region = get_region(mask,seed)
        mask[region[0],region[1]] = False
        regions[region[0],region[1]] = i
        i += 1
    return regions,i

def avg_hue(img,region):
    hue = img[:,:,0]
    return np.mean(hue[region])

def avg_sat(img,region):
    sat = img[:,:,1]
    return np.mean(sat[region])

def avg_val(img,region):
    val = img[:,:,2]
    return np.mean(val[region])

def avg_dx(img,region):
    diff_x = np.abs(img - np.roll(img,1,axis=0))
    return np.mean(diff_x[region])

def avg_dy(img,region):
    diff_y = np.abs(img - np.roll(img,1,axis=1))
    return np.mean(diff_y[region])

def var_hue(img,region):
    hue = img[:,:,0]
    return np.var(hue[region])

def var_sat(img,region):
    sat = img[:,:,1]
    return np.var(sat[region])

def var_val(img,region):
    val = img[:,:,2]
    return np.var(val[region])

def avg_hue_dx(img,region):
    hue = img[:,:,0]
    diff_x = np.abs(hue - np.roll(hue,1,axis=0))
    return np.mean(diff_x[region])

def avg_sat_dx(img,region):
    sat = img[:,:,1]
    diff_x = np.abs(hue - np.roll(sat,1,axis=0))
    return np.mean(diff_x[region])

def avg_val_dx(img,region):
    val = img[:,:,2]
    diff_x = np.abs(val - np.roll(val,1,axis=0))
    return np.mean(diff_x[region])

def avg_hue_dy(img,region):
    hue = img[:,:,0]
    diff_x = np.abs(hue - np.roll(hue,1,axis=1))
    return np.mean(diff_x[region])

def avg_sat_dy(img,region):
    sat = img[:,:,1]
    diff_x = np.abs(hue - np.roll(sat,1,axis=1))
    return np.mean(diff_x[region])

def avg_val_dy(img,region):
    val = img[:,:,2]
    diff_x = np.abs(val - np.roll(val,1,axis=1))
    return np.mean(diff_x[region])

def euclidean(a,b):
    return np.sum((a-b)**2)

def my_kmeans(vec,n,dist):
    rand = np.random.random_integers(0,len(vec)-1,n)
    centers = []
    for i in range(n):
        centers.append(vec[rand[i]])
    centers = np.array(centers)
    last_error = -1
    error = -2
    while error != last_error:
        sums = np.zeros((n,len(vec[0]))).astype(np.float64)
        qnt = np.zeros(n)
        last_error = error
        error = 0
        for el in vec:
            dists = [dist(el,cen) for cen in centers]
            error += np.min(dists)
            group = np.argmin(dists)
            sums[group] += el
            qnt[group] += 1
        for i in range(n):
            centers[i] = sums[i]/(qnt[i] + 0.0001 )
        #print(error)
    return centers

def rotate_hue(hsv):
    hue = hsv[:,:,0]
    hue = (hue + 0.5) % 1.0
    hsv[:,:,0] = hue
    return hsv


if __name__ == '__main__':
    for f in sys.argv[1:]:
        img = io.imread(f)
        hsv = color.rgb2hsv(img)
    
        hsv = rotate_hue(hsv)

        cols = less_colors(img)
        mask = cols == 1
        #mask = morphology.erosion(mask,morphology.square(5))
        mask = morphology.opening(mask,morphology.square(6))
        mask = morphology.closing(mask,morphology.square(4))
        regions,num = split_regions(mask)


        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        val = hsv[:,:,2]
        
        hue_dx = np.abs(hue - np.roll(hue,1,axis=0))
        sat_dx = np.abs(sat - np.roll(sat,1,axis=0))
        val_dx = np.abs(val - np.roll(val,1,axis=0))

        hue_dy = np.abs(hue - np.roll(hue,1,axis=1))
        sat_dy = np.abs(sat - np.roll(sat,1,axis=1))
        val_dy = np.abs(val - np.roll(val,1,axis=1))


        ds = []
        ms = [hue,hue_dx,hue_dy,sat,sat_dx,sat_dy,val,val_dx,val_dy]
        for i in range(1,num):
            d = [np.mean(m[regions == i]) for m in ms]
            d = np.array(d)
            ds.append(d)

        centers = my_kmeans(ds,4,euclidean)

        groups = np.zeros(mask.shape)
        for i in range(1,num):
            dists = [euclidean(ds[i-1],cen) for cen in centers]
            group = np.argmin(dists)
            groups[regions == i] = group + 1

        
        plt.imshow(groups)
        plt.show()
    
    

