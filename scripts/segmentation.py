from skimage.segmentation import slic
from skimage import color


def slic_segmentation(img):
    segments = slic(img, n_segments=3000, compactness=15)
    rgb_arr=color.label2rgb(segments, img, kind='avg')
    return rgb_arr
