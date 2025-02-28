import glob
import numpy as np
import cv2 as cv
import os
import random

from config import parse_cfg

def get_corners(top_left_obj, obj):
    
    bot_right_obj = (top_left_obj[0]+obj.shape[1], top_left_obj[1]+obj.shape[0])
    bot_left_obj = top_left_obj[0], bot_right_obj[1]
    top_right_obj = bot_right_obj[0], top_left_obj[1]    

    obj_corners = [top_left_obj, top_right_obj, bot_right_obj, bot_left_obj]
    
    return obj_corners

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
            

def resize(img, scalar):
    img = cv.resize(img, (int(img.shape[1]*scalar), int(img.shape[0]*scalar)))
    img = adaptive_thresh(img)
    
    return img
        
    
def adaptive_thresh(img, block_size=17, constant=10):
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
    
def get_scaled_position(position, old_size, new_size):
    x_p = position[0] / old_size[0]
    y_p = position[1] / old_size[1]

    return int(x_p*new_size[0]), int(y_p*new_size[1])

def get_img_data(directory_path):
    
    path_slash = '\\' if os.name == 'nt' else '/'
    
    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    img_names = []
    imgs = []
    
    for file_name in glob.glob(directory_path + '*.png'):

        cur_img = cv.imread(file_name)[:,:,0]
        imgs.append(cur_img)

        cur_img_name = file_name.split(path_slash)[-1]
        cur_img_name = cur_img_name.split('.')[0]         
        img_names.append(cur_img_name)
            
 
    return imgs


def get_number_of_images(directory_path):
    path_slash = '\\' if os.name == 'nt' else '/'

    if directory_path[-1] != path_slash:
        directory_path += path_slash
    
    return len(glob.glob(directory_path + '*.png'))


def delete_captured_images():
    
    for i in glob.glob('dots/*'):
        os.remove(i)

    for i in glob.glob('nums/*'):
        for j in glob.glob(i+'/*'):
            os.remove(j)

    for i in glob.glob('canvases/*'):
        os.remove(i)

    for i in glob.glob('backgrounds/*'):
        os.remove(i)

    for i in glob.glob('drawings/*'):
        os.remove(i)