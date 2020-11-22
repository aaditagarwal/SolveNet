import cv2
import keras
import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation.slic_superpixels import slic

from .processing import slic_segmentation
from .processing import extract_line
from .processing import text_segment
from .processing import evaluate

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
    result = list(df.apply(lambda row: text_segment(row['y1'],row['y2'],\
                 row['x1'],row['x2']), axis=1))

    df_chars = pd.DataFrame(result)

    char_area_list = []
    df_chars['char_df'].apply(lambda d: char_area_list.append(list(d['area'])))

    #Area based threshold for detecting and removing noises
    gamma = 0
    max_ar = max([max(i) for i in char_area_list])
    ar_thresh = max_ar*gamma

    #Keeping only those characters whose area of contours is above area threshold
    df_chars['char_df'] = df_chars['char_df'].apply(lambda d: d[d.area > ar_thresh])

    box_img = dict_clean_img['r'] #For Processing B/W image
    box_img_1 = dict_img['r'] #For saving results
    box_img = cv2.cvtColor(box_img, cv2.COLOR_GRAY2BGR)

    df = df_chars.copy()
        
    df['char_df'].apply(lambda d: d.apply(lambda c: cv2.rectangle(box_img, \
        (c['X1'],c['Y1']),(c['X2'], c['Y2']),(255*(c['exp']==1),180,0),2+(2*c['exp'])), axis=1 ) )
    
    df['line_status'] = df['char_df'].apply(lambda d: evaluate(d[["pred","exp","pred_proba"]]))
    
    return df