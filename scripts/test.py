import cv2
import keras
import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation.slic_superpixels import slic

from processing import slic_segmentation
from processing import extract_line
from processing import text_segment
from processing import evaluate

from predict import run

run("D:/123.png")