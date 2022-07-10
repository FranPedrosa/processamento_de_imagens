from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

import split_borders


def coke_reds(hsv):
    sat = hsv[:,:,1] > 0.15
    hue = abs(hsv[:,:,0] - 0.5) < 0.05
    return np.logical_and(sat,hue)

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


def rotate_hue(hsv):
    hue = hsv[:,:,0]
    hue = (hue + 0.5) % 1.0
    hsv[:,:,0] = hue
    return hsv

def euclidean(a,b):
    d2 = np.sum((a-b)**2)
    return math.sqrt(d2)

def polar(shape):
    center = np.mean(shape,axis=0)
    rads = [ euclidean(center,p) for p in shape]
    norm = np.array(rads) / np.max(rads)
    return np.roll(norm,-np.argmin(norm))

def sample(vetor,n):
    return [ np.mean(seg) for seg in np.array_split(vetor,n)]


if __name__ == '__main__':

    f = open('a.out','rb')
    ref = np.load(f)
    print(ref.shape)

    for f in sys.argv[1:]:
        print(f)
        img = io.imread(f)
        hsv = color.rgb2hsv(img)
    
        hsv = rotate_hue(hsv)

        mask = coke_reds(hsv)
        mask = morphology.opening(mask,morphology.square(8))
        mask = np.pad(mask,1)

        shapes = split_borders.split(mask,40)
        polars = [ polar(s) for s in shapes]
        samples = [sample(p,40) for p in polars]

        '''
        i = 0
        for s in samples:
            print(np.sum(np.abs(s-ref)))
            plt.subplot(121)
            plt.plot(s,'b-')
            plt.plot(ref,'g-')
            plt.plot(np.abs(ref-s),'r-')

            cp = img.copy()
            for p in shapes[i]:
                cp[p] = [0,255,0]
            plt.subplot(122)
            plt.imshow(cp)
            plt.show()
            i+=1
        '''
            


        img = np.pad(img,((1,1),(1,1),(0,0)))
        dists = np.zeros(mask.shape)
        i = 0
        for s in samples:
            d = np.sum(np.abs(s-ref))
            if d < 3:
                for p in shapes[i]:
                    img[p] = [0,255,0]
            else:
                for p in shapes[i]:
                    img[p] = [0,0,255]
            i+=1
            
        plt.subplot(111)
        plt.imshow(img)
        plt.show()
            


    
    

