from skimage import morphology
from skimage.morphology import disk

rectangle = [[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]

def caps(mask):
    mask = morphology.erosion(mask,rectangle)
    mask = morphology.dilation(mask,rectangle)
    mask = morphology.dilation(mask,morphology.square(8))
    mask = morphology.erosion(mask,morphology.square(8))
    return mask

def filter(mask):
    mask = morphology.erosion(mask,disk(2))
    mask = morphology.dilation(mask,disk(2))
    mask = morphology.dilation(mask,disk(5))
    mask = morphology.erosion(mask,disk(5))
    return mask

