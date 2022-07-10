from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys

import split_borders


def coke_reds(hsv):
    sat = hsv[:,:,1] > 0.15
    hue = hsv[:,:,0] - 0.5 < 0.05
    return np.logical_and(sat,hue)

def neighbors_4(center):
    x,y = center[0],center[1]
    return (x+1,y),(x,y-1),(x-1,y),(x,y+1)


def valid(p,w,h):
    return 0 <= p[0] and p[0] < w and 0 <= p[1] and p[1] < h 

def euclidean(a,b):
    return math.sqrt(np.sum((a-b)**2))

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


def rotate_hue(hsv):
    hue = hsv[:,:,0]
    hue = (hue + 0.5) % 1.0
    hsv[:,:,0] = hue
    return hsv


def polar(shape):
    center = np.mean(shape,axis=0)
    rads = [ euclidean(center,p) for p in shape]
    norm = np.array(rads) / np.max(rads)
    return np.roll(norm,-np.argmin(norm))

def sample(vetor,n):
    return [ np.mean(seg) for seg in np.array_split(vetor,n)]

def my_kmeans(vec,n,dist):
    centers = random.sample(list(vec),n)
    centers = np.array(centers)
    last_error = -1
    error = -2
    while error != last_error:
        sums = np.zeros((n,len(vec[0]))).astype(np.complex128)
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
    groups = np.zeros((vec.shape[0]))
    i = 0
    for el in vec:
        dists = [dist(el,cen) for cen in centers]
        groups[i] = np.argmin(dists)
        i+=1
    return centers


if __name__ == '__main__':
    ds = []
    for f in sys.argv[1:]:
        print(f)
        img = io.imread(f)
        hsv = color.rgb2hsv(img)
    
        hsv = rotate_hue(hsv)

        mask = coke_reds(hsv)
        mask = morphology.opening(mask,morphology.square(8))
        mask = np.pad(mask,1)

        plt.imshow(mask)
        plt.show()

        shapes = split_borders.split(mask,40)
        polars = [ polar(s) for s in shapes]
        samples = [sample(p,40) for p in polars]

        ds += samples


    ds = np.array(ds)
    means = np.mean(ds,axis=0)
    f = open('a.out','wb')
    np.save(f,means)

    
    

