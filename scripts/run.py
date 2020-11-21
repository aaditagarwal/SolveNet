import numpy as np
import pandas as pd
import keras
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation.slic_superpixels import slic

from processing import slic_segmentation
from processing import extract_line
from processing import text_segment

keras.backend.set_image_data_format("channels_first")

def run(img_source):
    source_img = img_as_float(io.imread(img_source))
    img = slic_segmentation(source_img)
    img = img.astype(np.uint8)

    #Global Variable
    dict_clean_img = {} #BINARY IMAGE DICTIONAY
    dict_img = {} #ORIGINAL IMAGE DICTIONARY
    df_lines = pd.DataFrame()

    #Extracting lines present in the boxes
    H,W = img.shape[:2]
    cleaned_orig,y1s,y2s = extract_line(img)
    x1s = [0]*len(y1s)
    x2s = [W]*len(y1s)

    df = pd.DataFrame([y1s,y2s,x1s,x2s]).transpose()
    df.columns = ['y1','y2','x1','x2']

    df_lines= pd.concat([df_lines, df])

    dict_clean_img.update({"r":cleaned_orig})
    dict_img.update({"r":img})

    #df_chars contains locations of all characters along with box_num and line name
    result = df.apply(lambda row: text_segment(row['y1'],row['y2'],\
                 row['x1'],row['x2']), axis=1, result_type='broadcast')

    return result