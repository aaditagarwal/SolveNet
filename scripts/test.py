import cv2
import numpy as np
import pandas as pd
import time

def run(direc): 
	ans = "75"
	#src = cv2.imread(path) 
	#img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY ) 
	df = pd.DataFrame(np.random.rand(10,10))
	print("DIREC:"+direc)
	img = direc
	return df, ans, img