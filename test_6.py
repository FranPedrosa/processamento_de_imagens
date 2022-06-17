from skimage import io, color, morphology, filters
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from split_borders import split

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
    return np.sum((a-b)**2)

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


def blue_caps(src):
    img = src.copy()
    #b = color.rgb2lab(img)[:,:,2]
    #b = normalize(b)
    #mask = multiple_otsu(b,2)

    v = color.rgb2hsv(img)[:,:,1]
    v = normalize(v)
    mask = multiple_otsu(v,1)

    mask = morphology.erosion(mask, morphology.disk(1))
    mask = morphology.dilation(mask, morphology.disk(5))
    mask = morphology.erosion(mask, morphology.square(5))
    mask = morphology.dilation(mask, morphology.disk(5))
    mask = morphology.erosion(mask, morphology.disk(9))

    paths = split(mask)
    
    fds = []
    
    fig = 0
    for pa in paths:
        zeros = np.zeros(mask.shape)
        tans = []
        for i in range(0,len(pa)):
            j = (i - 1) % len(pa)
            tan = angle(pa[i],pa[j])
            if tan == 0 and last_tan > 4:
                tan = 8
            last_tan = tan
            tans.append(tan)
        #ker = np.ones(int(10))/10
        #longer = np.concatenate([tans[-int(10):],tans])
        #tans = np.convolve(longer,ker,'valid')
        #tans = np.roll(tans,-np.argmin(tans))
        #xs = np.linspace(0,8,len(tans))
        #tans -= xs

        samples = [ int(2*np.mean(seg)) for seg in np.array_split(tans,10)]
        longer = np.append(samples,samples[0])

        ker = [-1,1]
        der = np.convolve(longer,ker,'valid')
        der = np.roll(der,-np.argmax(der))


        #fd = np.fft.fft(der)
        #fd = np.concatenate((fd.real,fd.imag))

        #w = len(tans)/10
        #samples = []
        #for seg in np.array_split(der,10):
        #    pos = np.argmax(np.abs(seg))
        #    samples.append(seg[pos])


        fds.append(der)

        #fd = np.fft.fft(tans,n=12).real
        #fds.append(fd)

        #ifd = np.fft.ifft(fd,n=len(tans)).real
            
        plt.subplot(131)
        plt.plot(tans)
        plt.subplot(132)
        plt.plot(samples)
        plt.subplot(133)
        plt.plot(der)
        plt.show()
        #plt.savefig('graphs/'+str(i)+'.png')
        fig+=1
    

    fds = np.array(fds)

    result = np.mean(fds,axis=0)
    print(result)

    dists = [euclidean(a,result) for a  in fds]
    print('euclidean: ', dists)

    dists = [manhattam(a,result) for a  in fds]
    print('manhattam: ', dists)

    dists = [relative(a,result) for a  in fds]
    print('relative: ', dists)

    plt.legend([str(i) for i in range(12)])
    plt.show()

    #f = open(sys.argv[2],'wb')
    #np.save(f,result)

    #plt.imshow(dists,cmap='gray')
    #plt.show()

    colors = [[255,0,0],[255,255,0],[0,255,0],[0,255,255],[0,0,255],[255,0,255],[255,255,255],[125,125,125]]
    black = np.zeros(img.shape)


    #centers = my_kmeans(fds,8,euclidean)
    groups = similarity_tree(fds,3,relative) 

    i = 0
    for pa in paths:
        #dists = [euclidean(fds[i],cen) for cen in centers]
        #group = np.argmin(dists)
        group = groups[i]
        for p in pa:
            black[p] = colors[group]
        i += 1

    plt.imshow(black)
    plt.show()

    


for f in [sys.argv[1]]:
    print(f)
    src = io.imread(f)
    img = src.copy()
    
    blue_caps(img)
