from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, color
import matplotlib.pyplot as plt
import skimage.segmentation as seg

def slic_segmentation(img):
    segments = slic(img, n_segments=1000, compactness=10)
    rgb_arr=color.label2rgb(segments, img, kind='avg')
    return rgb_arr

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax