import numpy as np
import glob
import os

from config_utils import parse_cfg

def get_blanks(directory_path):
    
    # clean after using numbers to name images
    path_slash = '\\' if os.name == 'nt' else '/'
        
    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    blank_names = set()
    
    for file_name in glob.glob(directory_path + '*.npy'):
        cur_blank_name = file_name.split(path_slash)[-1]
        cur_blank_name = cur_blank_name.split('.')[0]
        
        blank_names.add(cur_blank_name)
        
    cfg_file = glob.glob(directory_path + '*.cfg')[0]
    cfg =  parse_cfg(cfg_file)

    blanks = []

    for blank in blank_names:
        img = np.load(directory_path + blank + '.npy')
        top_left = [int(x) for x in cfg[blank]['top_left']]
        bot_right = [int(x) for x in cfg[blank]['bot_right']]

        
        blanks.append(Blank(img, top_left, bot_right))
        
    return blanks   


def write_to_cfg(blanks, append=True):
    
    mode = 'a' if append else 'w+'
    
    with open('blanks/blank.cfg', mode=mode) as f:
        sorted_keys = sorted(blanks)
        
        for key in sorted_keys:
            min_row, min_col = blanks[key][0]
            max_row, max_col = blanks[key][1]
       
            f.writelines('\n[%d]' % key)
            f.writelines('\ntop_left =  %d, %d' % (min_row, min_col))
            f.writelines('\nbot_right =  %d, %d' % (max_row, max_col))
            f.writelines('\n[end]\n')
            
            
class Blank:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        self.top_left = top_left
        self.bot_right = bot_right        
        
        self.min_row, self.max_row, self.min_col, self.max_col = top_left[0], bot_right[0], top_left[1], bot_right[1]

        self.can_place = np.zeros(shape=img.shape)
        
            
    def place_object(self, obj, top_left_obj):
        """ tries to place an object(image) on the current blank canvas
        
            Parameters
            ----------
            obj : numpy array
                    image object to be drawn on top of blank
            
            top_left_obj : tuple (row, col)
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
       
        min_row, max_row, min_col, max_col = top_left_obj[0], bot_right_obj[0], top_left_obj[1], bot_right_obj[1]   
        
        if np.max(self.can_place[min_row:max_row, min_col:max_col]) == 1:
            return False

        self.can_place[min_row:max_row, min_col:max_col]  = 1
        self.img[min_row:max_row, min_col:max_col] = obj
        
        return True
        
                                     
        