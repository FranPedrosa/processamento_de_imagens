from skimage import io, color
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
    img = io.imread(f)
    img_2 = io.imread(f)

    sat = color.rgb2hsv(img)[:,:,1]
    mask = otsu_segmentation(sat)

    img[mask] = [0,0,0]
    dst = '../otsu_sat/'+f
    io.imsave(dst,img)

    inv_mask = np.logical_not(mask)
    img_2[inv_mask] = [0,0,0]
    dst = '../otsu_sat_rev/'+f
    io.imsave(dst,img_2)

    #plt.imshow(img_2)
    #plt.show()
