import glob
import numpy as np
import cv2 as cv
import os

def jpg_to_numpy(directory_path, delete=False):
    
    for img_path in glob.glob(directory_path + '/' + '*.jpg'):
        
        img = cv.imread(img_path)
        np.save(img_path.replace('.jpg', ''), img)
        
        if delete:
            os.remove(img_path)
            

