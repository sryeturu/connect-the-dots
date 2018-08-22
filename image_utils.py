import glob
import numpy as np
import cv2 as cv
import os

from config_utils import parse_cfg


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
        
    min_row, max_row, min_col, max_col = top_left[0], bot_right[0], top_left[1], bot_right[1]   
    
    np.save(save_path, img[min_row:max_row, min_col:max_col])
    
    
def get_img_data(directory_path):
    
    path_slash = '\\' if os.name == 'nt' else '/'
    
    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    img_names = []
    imgs = []
    
    for file_name in glob.glob(directory_path + '*'):

        cur_img = np.load(file_name)
        imgs.append(cur_img)

        cur_img_name = file_name.split(path_slash)[-1]
        cur_img_name = cur_img_name.split('.')[0]         
        img_names.append(cur_img_name)
            
 
    return img_names, imgs


def get_number_of_images(directory_path):
    path_slash = '\\' if os.name == 'nt' else '/'

    if directory_path[-1] != path_slash:
        directory_path += path_slash
    
    return len(glob.glob(directory_path + '*.npy'))