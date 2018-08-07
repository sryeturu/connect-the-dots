import numpy as np
import glob
import os

from image_utils import get_min_max_coords

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
    
    path_slash = '\\' if os.name == 'nt' else '/'
    
    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    blank_names = set()
    
    for file_name in glob.glob(directory_path + '*'):
                
        cur_blank_name = file_name.split(path_slash)[1]
        cur_blank_name = cur_blank_name.split('.')[0]
        
        blank_names.add(cur_blank_name)
        
    blanks = []
    
    for blank in blank_names:
        img = np.load(directory_path + blank + '.npy')
        top_left, bot_right = parse_blank_fields(directory_path + blank + '.txt')
        
        blanks.append(Blank(img, top_left, bot_right))
        
    return blanks    


class Blank:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        self.top_left = top_left
        self.bot_right = bot_right        
        
        self.min_row, self.max_row, self.min_col, self.max_col = get_min_max_coords(top_left, bot_right)

        self.can_place = np.zeros(shape=img.shape)
        
            
    def place_object(self, obj, top_left_obj):
        """ tries to place an object(image) on the current blank canvas
        
            Parameters
            ----------
            obj : numpy array
                    image object to be drawn on top of blank
            
            top_left_coords : tuple (row, col)
                    the row and column of where the top left corner of the image should be placed
            
            Returns
            ----------
            bool
                wether the placement was succesful or not
        """
        
        bot_right_obj = (top_left_obj[0]+obj.shape[0], top_left_obj[1]+obj.shape[1])
        
        if top_left_obj[0] < self.top_left[0]:
            return False
        if top_left_obj[1] < self.top_left[1]:
            return False
        if bot_right_obj[0] > self.bot_right[0]:
            return False
        if bot_right_obj[1] > self.bot_right[1]:
            return False
       
        min_row, max_row , min_col, max_col = get_min_max_coords(top_left_obj, bot_right_obj)
        
        if np.max(self.can_place[min_row:max_row, min_col:max_col]) == 1:
            return False

        self.can_place[min_row:max_row, min_col:max_col]  = 1
        self.img[min_row:max_row, min_col:max_col] = obj
        
        return True
        
                                     
        