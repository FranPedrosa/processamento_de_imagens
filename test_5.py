from skimage import io, color, morphology, filters
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def blue_caps(src):
    img = src.copy()
    b = color.rgb2lab(img)[:,:,2]
    b = normalize(b)
    mask = multiple_otsu(b,2)

    diagonal_1 = [
        [0,0,0,0,0,0,1],
        [0,0,0,0,0,1,0],
        [0,0,0,0,1,0,0],
        [0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0],
        [0,1,0,0,0,0,0],
        [1,0,0,0,0,0,0]]

    diagonal_2 = [
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]]

    plt.imshow(mask)
    plt.show()
    mask = morphology.erosion(mask, morphology.disk(1))
    mask = morphology.dilation(mask, morphology.disk(5))
    mask = morphology.erosion(mask, morphology.square(5))
    plt.imshow(mask)
    plt.show()
    mask = morphology.dilation(mask, morphology.disk(5))
    mask = morphology.erosion(mask, morphology.square(5))
    plt.imshow(mask)
    plt.show()
    mask = morphology.erosion(mask, morphology.square(8))
    plt.imshow(mask)
    plt.show()
    mask = morphology.erosion(mask, diagonal_1)
    plt.imshow(mask)
    plt.show()
    mask = morphology.erosion(mask, diagonal_2)
    #mask = morphology.dilation(mask, morphology.disk(5))
    #mask = morphology.erosion(mask, morphology.square(5))
    #mask = morphology.erosion(mask, morphology.square(8))

    #mask = mask.astype(np.uint8)
    #lap = filters.laplace(mask)

    plt.imshow(mask)
    plt.show()
    


for f in sys.argv[1:]:
    print(f)
    src = io.imread(f)
    img = src.copy()
    
    blue_caps(img)
