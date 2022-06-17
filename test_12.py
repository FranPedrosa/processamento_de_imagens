from skimage import io, color, filters
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from split_borders import split
from split_colors import *
import morphology

def hist_mean(hist,tam,ini):
    i = ini
    soma = 0
    for j in hist:
        soma += i*j
        i += 1
    return soma/(tam+1e-10)

def hist_var(hist,tam,ini):
    m = hist_mean(hist,tam,ini)
    i = ini
    soma = 0
    for j in hist:
        soma += (i - m)**2 * j
        i += 1
    return soma/ (tam+1e-10)

def normalize(img):
    top = np.max(img)
    bottom = np.min(img)
    return (img - bottom)/(top-bottom)

def otsu_variance(hist,l,tam):
    a = hist[:l]
    b = hist[l:]
    peso_a = np.sum(a)
    peso_b = np.sum(b)
    var_a = hist_var(a,peso_a,0)
    var_b = hist_var(b,peso_b,l)
    peso_a /= tam
    peso_b /= tam
    return peso_a*var_a + peso_b*var_b

def otsu_segmentation(img):
    top = np.max(img)
    hist = np.histogram(img,bins=100,range=(0,top))[0]
    tam = sum(hist)
    todas_var = [ otsu_variance(hist,l,tam) for l in range(1,100)]
    l = (np.argmin(todas_var)+1) / 100 * top
    return img < l

def multiple_otsu(img,n):
    step = img.copy()
    for i in range(n):
        top = np.max(step)
        hist = np.histogram(img,bins=100,range=(0,top))[0]
        tam = sum(hist)
        todas_var = [ otsu_variance(hist,l,tam) for l in range(1,100)]
        l = (np.argmin(todas_var)+1) / 100 * top
        step = step[step < l]
    return img < l


def euclidean(a,b):
    return np.sum((a-b)**2)/len(a)

def manhattam(a,b):
    return np.sum(np.abs((a-b)))

def relative(a,b):
    return np.sum(np.abs((a-b)/(a+b+1)))

def wierd(a,b):
    return np.var(a-b)


def my_kmeans(vec,n,dist):
    rand = np.random.random_integers(0,len(vec)-1,n)
    print(rand)
    centers = []
    for i in range(n):
        centers.append(vec[rand[i]])
    centers = np.array(centers)
    print(centers)
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
        #print(error)
    return centers

def similarity_tree(vec,n,dist):
    qnt = len(vec)
    groups = np.array(range(qnt))
    aux = vec.copy()
    dists = np.array([[ dist(a,b) for a  in aux] for b in aux])
    for i in range(qnt):
        dists[i,i] = math.inf
    while qnt > n:
        print(groups)
        pos = np.argmin(dists)
        a,b = pos//qnt,pos%qnt
        mean = (aux[a]+aux[b])/2
        aux[a] = mean
        dists[a,:] = np.array([ dist(el,mean) for el in aux])
        dists[a,a] = math.inf
        groups[groups == b] = a
        groups[groups > b] -= 1
        aux = np.delete(aux,b,0)
        dists = np.delete(dists,b,0)
        dists = np.delete(dists,b,1)
        qnt -= 1
    return groups

def angle(a,b):
    dy = a[0] - b[0] + 1
    dx = a[1] - b[1] + 1
    op = dy*3 + dx
    angs = [3,2,1,4,None,0,5,6,7]
    return angs[op] 


def tangents(pa,size):
    tans = []
    last = angle(pa[-1],pa[0])
    for i in range(0,len(pa)):
        j = (i - 1) % len(pa)
        tan = angle(pa[i],pa[j])
        #if abs(last - tan) < 2 or abs(last - tan) > 6:
        #    tan = last 
        last = tan
        tans.append(tan)
    tans = np.roll(tans,-np.argmin(tans))
    samples = [ np.mean(seg) for seg in np.array_split(tans,size)]
    return np.array(samples)


def count_shapes(paths,know_shapes,limit):
    count = np.zeros(len(paths))
    tans = [ tangents(pa) for pa in paths]

    for t in tans:
        dists = [euclidean(t,shape) for shape in know_shapes]
        if min(dists) < limit:
            meaning = np.argmin(dists)
            count[meaning] += 1

    return count


def train_shapes(shapes,qnt_shapes,tans,limit):
    dists = [euclidean(tans,s) for s in shapes]
    if len(shapes) > 0 and min(dists) < limit:
        i = np.argmin(dists)
        shapes[i] = (qnt_shapes[i]*shapes[i] + tans)/(qnt_shapes[i]+1)
        qnt_shapes[i] += 1
    else:
        shapes.append(tans)
        qnt_shapes.append(1)
    return shapes,qnt_shapes

def recognize(tans,pattern,limit):
    ret = []
    for i in range(len(tans)):
        t = tans[i]
        dist = euclidean(t,pattern)
        ret.append(dist < limit)
    return ret

def overlay(src,paths):
    img = src.copy()
    w,h,c = img.shape
    padded = np.zeros((w+2,h+2,c)).astype(int)
    for x in range(w):
        for y in range(h):
            padded[x+1,y+1] = src[x,y]
    for path in paths:
        for p in path:
            padded[p] = [255,0,255]
    plt.imshow(padded)
    plt.show()

if __name__ == '__main__':
    img = io.imread(sys.argv[1])

    a = less_colors(img)

    patterns,qnt = [],[]
    shapes = []
    tans = []

    w,h,_ = img.shape
    black = np.zeros((w+2,h+2))
    
    for col in range(1,9):
        mask = a == col
        mask = morphology.filter(mask)
        mask = np.pad(mask,1)
        paths = split(mask,20)
        shapes += paths

    for path in shapes:
        for p in path:
            black[p] = 1
            

    plt.imshow(black)
    plt.show()

    overlay(img,shapes)
        
        
