import glob
import numpy as np
import cv2 as cv
import os

from blank import Blank

def jpg_to_numpy(directory_path, gray=True, delete=False):
    
    if directory_path[-1] != '/':
        directory_path += '/'
        
    for img_path in glob.glob(directory_path + '/' + '*.jpg'):
        
        img = cv.imread(img_path)
        
        if gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        np.save(img_path.replace('.jpg', ''), img)
        
        if delete:
            os.remove(img_path)
            

def adaptive_thresh(img, block_size=45, constant=10):
    return cv.adaptiveThreshold(img, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant) 


def parse_blank_fields(filename):
    
    fields = {}
    
    with open(filename) as f:
        for line in f:
            line = line.split(':')
            
            field_name = line[0].strip()
            
            coordinates = line[1].split(',')
            row = int(coordinates[0].strip())
            col = int(coordinates[1].strip())
            
            fields[field_name] = row,col
    
    return fields['top_left'], fields['bot_right']


def get_blanks(directory_path):
    
    if directory_path[-1] != '/':
        directory_path += '/'
        
    blank_names = set()
    
    for file_name in glob.glob(directory_path + '/*'):
        cur_blank_name = file_name.split('/')[1]
        cur_blank_name = cur_blank_name.split('.')[0]
        
        blank_names.add(cur_blank_name)
        
    blanks = []
    
    for blank in blank_names:
        img = np.load(directory_path + '/' + blank + '.npy')
        top_left, bot_right = parse_blank_fields(directory_path + '/' + blank + '.txt')
        
        blanks.append(Blank(img, top_left, bot_right))
        
    return blanks