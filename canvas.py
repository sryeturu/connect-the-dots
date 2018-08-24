import numpy as np
import glob
import os
from image_utils import get_number_of_images
from config import parse_cfg

def get_canvases(directory_path):
    
    path_slash = '\\' if os.name == 'nt' else '/'
        
    if directory_path[-1] != path_slash:
        directory_path += path_slash
        
    cfg_file = glob.glob(directory_path + '*.cfg')[0]
    cfg =  parse_cfg(cfg_file)

    canvases = []
    
    num_of_canvases = get_number_of_images(directory_path)
    for i in range(1, num_of_canvases+1):
        i = str(i)
        img = np.load(directory_path + i + '.npy')
        top_left = [int(x) for x in cfg[i]['top_left']]
        bot_right = [int(x) for x in cfg[i]['bot_right']]   

        
        canvases.append(Canvas(img, top_left, bot_right))
        
    return canvases   


def write_to_cfg(canvases, append=True):
    
    mode = 'a' if append else 'w+'
    
    with open('canvases/canvas.cfg', mode=mode) as f:
        sorted_keys = sorted(canvases)
        
        for key in sorted_keys:
            min_row, min_col = canvases[key][0]
            max_row, max_col = canvases[key][1]
       
            f.writelines('\n[%d]' % key)
            f.writelines('\ntop_left =  %d, %d' % (min_row, min_col))
            f.writelines('\nbot_right =  %d, %d' % (max_row, max_col))
            f.writelines('\n[end]\n')
            
            
class Canvas:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        # specify these are paper attributes
        self.top_left = top_left
        self.bot_right = bot_right        
        
        self.min_row, self.max_row, self.min_col, self.max_col = top_left[0], bot_right[0], top_left[1], bot_right[1]

        self.can_place = np.zeros_like(img)
        self.can_place[np.where(img == 0)] = 1 # where it's already black(hand, page outline) we CANNOT draw on
        
    
    def draw_on_background(self, obj, top_left_obj):
        
        for row in range(obj.shape[0]):        
            if top_left_obj[0] + row >= self.img.shape[0]:
                break
                
            for col in range(obj.shape[1]):

                if top_left_obj[1] + col >= self.img.shape[1]:
                    break
                
                if (self.min_row <= (top_left_obj[0] + row) <= self.max_row) and (self.min_col <= (top_left_obj[1] + col) <= self.max_col):
                    continue

                self.img[top_left_obj[0] + row, top_left_obj[1] + col] = obj[row, col]

    def draw_on_paper(self, obj, top_left_obj):
        """ tries to place an object(image) on the canvas paper
        
            Parameters
            ----------
            obj : numpy array
                    image object to be drawn on top of canvas paper
            
            top_left_obj : tuple (row, col)
                    the row and column of where the top left corner of the image should be placed
            
            Returns
            ----------
            bool
                wether the placement was succesful or not
        """
        
        bot_right_obj = (top_left_obj[0]+obj.shape[0], top_left_obj[1]+obj.shape[1])
        min_row, max_row, min_col, max_col = top_left_obj[0], bot_right_obj[0], top_left_obj[1], bot_right_obj[1]   

        if top_left_obj[0] < self.top_left[0]:
            return False
        if top_left_obj[1] < self.top_left[1]:
            return False
        if bot_right_obj[0] > self.bot_right[0]:
            return False
        if bot_right_obj[1] > self.bot_right[1]:
            return False
       
        if np.max(self.can_place[min_row:max_row, min_col:max_col]) == 1:
            return False

        self.can_place[min_row:max_row, min_col:max_col]  = 1
        self.img[min_row:max_row, min_col:max_col] = obj
        
        return True
        
                                     
        