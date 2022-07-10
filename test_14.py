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


if __name__ == '__main__':
    img = io.imread(sys.argv[1])
    hsv = color.rgb2hsv(img)

    lap = filters.laplace(hsv)

    cols = less_colors(img)
    mask = cols == 1
    mask = morphology.erosion(mask,morphology.square(4))
    #mask = morphology.opening(mask,morphology.square(5))
    #mask = morphology.closing(mask,morphology.square(3))
    regions,num = split_regions(mask)

    ds = []
    for i in range(1,num):
        vals = hsv[regions == i]
        means = np.mean(vals,axis=0)
        ders = lap[regions == i]
        unis = np.mean(ders,axis=0)
        ds.append(np.concatenate([means,unis]))

    centers = my_kmeans(ds,2,euclidean)

    groups = np.zeros(mask.shape)
    for i in range(1,num):
        dists = [euclidean(ds[i-1],cen) for cen in centers]
        group = np.argmin(dists)
        groups[regions == i] = group + 1
    
    plt.imshow(groups)
    plt.show()
    
    

