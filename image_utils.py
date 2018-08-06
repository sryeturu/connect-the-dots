import glob
import numpy as np
import cv2 as cv
import os

def jpg_to_numpy(directory_path, gray=True, delete=False):
    
    for img_path in glob.glob(directory_path + '/' + '*.jpg'):
        
        img = cv.imread(img_path)
        
        if gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        np.save(img_path.replace('.jpg', ''), img)
        
        if delete:
            os.remove(img_path)
            

def adaptive_thresh(img, block_size=45, constant=10):
    return cv.adaptiveThreshold(img, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant) 