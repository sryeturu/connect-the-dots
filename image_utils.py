import glob
import numpy as np
import cv2 as cv
import os

def jpg_to_numpy(directory_path, gray=True, delete=False):
    
    path_slash = '\\' if os.name == 'nt' else '/'

    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    for img_path in glob.glob(directory_path + '*.jpg'):              
        img = cv.imread(img_path)
        
        if gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        np.save(img_path.replace('.jpg', ''), img)
        
        if delete:
            os.remove(img_path)
            

def adaptive_thresh(img, block_size=45, constant=10):
    return cv.adaptiveThreshold(img, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant) 


def cut_and_save(img_path, save_path, top_left, bot_right, gray=True):
    
    """ 
    img file should be kept in current directory. saved file will be saved there as well.
    """
    
    img = cv.imread(img_path)
        
    if gray:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    min_row, max_row , min_col, max_col = get_min_max_coords(top_left, bot_right)
    
    np.save(save_path, img[min_row:max_row, min_col:max_col])
    

def get_min_max_coords(top_left, bot_right):
    """ this function returns the min and max values
        for the row and column given the top left and bottom right corner coordinates. 
        
    """           
    min_row = top_left[0]
    max_row = bot_right[0]
    min_col = top_left[1]
    max_col = bot_right[1]
    
    return min_row, max_row, min_col, max_col
