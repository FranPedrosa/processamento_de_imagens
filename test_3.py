from skimage import io, color, morphology
import matplotlib.pyplot as plt
import numpy as np
import sys

def hist_mean(hist,tam,ini):
    i = ini
    soma = 0
    for j in hist:
        soma += i*j
        i += 1
    return soma/tam

def hist_var(hist,tam,ini):
    m = hist_mean(hist,tam,ini)
    i = ini
    soma = 0
    for j in hist:
        soma += (i - m)**2 * j
        i += 1
    return soma/tam
    

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
    hist = np.histogram(img,bins=100,range=(0,1))[0]
    tam = sum(hist)
    todas_var = [ otsu_variance(hist,l,tam) for l in range(1,100)]
    l = (np.argmin(todas_var)+1) / 100 
    return img < l

for f in sys.argv[1:]:
    print(f)
    src = io.imread(f)
    img = src.copy()
    img2 = src.copy()

    sat = color.rgb2hsv(img)[:,:,1]
    mask = otsu_segmentation(sat)
    img[mask] = [0,255,0]

    anti_mask = np.logical_not(mask)
    anti_mask = morphology.erosion(anti_mask, morphology.square(2))
    anti_mask = morphology.dilation(anti_mask, morphology.square(2))
    anti_mask = morphology.dilation(anti_mask, morphology.disk(6))
    anti_mask = morphology.erosion(anti_mask, morphology.disk(6))
    mask = np.logical_not(anti_mask)
    img2[mask] = [0,255,0]

    #dst = '../otsu_sat/'+f
    #io.imsave(dst,img)

    
    #plt.subplot(131)
    #plt.imshow(src)
    #plt.subplot(132)
    #plt.imshow(img)
    #plt.subplot(133)
    plt.imshow(img2)
    plt.show()
