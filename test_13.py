from skimage import io, color, filters
from skimage.morphology import square
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def neighbors(center):
    x,y = center[0],center[1]
    return (x+1,y),(x+1,y-1),(x,y-1),(x-1,y-1),(x-1,y),(x-1,y+1),(x,y+1),(x+1,y+1)

def lbp(img,center):
    comp = [img[ns] > img[center] for ns in neighbors(center)]
    changes = 0
    for i in range(8):
        j = ( i + 1) % 8
        if comp[i] != comp[j]:
            changes += 1
    return changes

def binary_patterns(src):
    w,h = src.shape
    dst = np.zeros((w-2,h-2))
    for y in range(1,h-1):
        for x in range(1,w-1):
            dst[x-1,y-1] = lbp(src,(x,y))
    return dst

if __name__ == '__main__':
    img = io.imread(sys.argv[1])

    hsv = color.rgb2hsv(img)

    h = hsv[:,:,0]
    s = hsv[:,:,1]
    
    print(h.shape)

    a = binary_patterns(h)
    a = convolve2d(a,square(20)/400)
    a = np.rint(2*a)

    b = binary_patterns(s)
    b = convolve2d(b,square(20)/400)
    b = np.rint(2*b)

    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
        
        
