from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import sys

def color_spaces(rgb):
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    hsv = color.rgb2hsv(rgb)

    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    lab = color.rgb2lab(rgb)

    l = lab[:,:,0]
    a = lab[:,:,1]
    B = lab[:,:,2]

    print(r.shape)
    print(type(r[0,0]))

    plt.subplot(331)
    plt.imshow(r,cmap='gray')
    plt.subplot(332)
    plt.imshow(g,cmap='gray')
    plt.subplot(333)
    plt.imshow(b,cmap='gray')

    plt.subplot(334)
    plt.imshow(h,cmap='gray')
    plt.subplot(335)
    plt.imshow(s,cmap='gray')
    plt.subplot(336)
    plt.imshow(v,cmap='gray')

    plt.subplot(337)
    plt.imshow(l,cmap='gray')
    plt.subplot(338)
    plt.imshow(a,cmap='gray')
    plt.subplot(339)
    plt.imshow(B,cmap='gray')

    plt.show()


rgb = io.imread(sys.argv[1])
color_spaces(rgb)
