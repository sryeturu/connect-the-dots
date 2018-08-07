import numpy as np

class Blank:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        self.top_left = top_left
        self.bot_right = bot_right
        
        self.min_row = top_left[0]
        self.max_row = bot_right[0]
                                 
        self.min_col = top_left[1]
        self.max_col = bot_right[1]
        
        self.can_place = np.zeros(shape=img.shape)
        
            
    def place_object(self, obj, top_left_obj):
        """ tries to place an object(image) on the current blank canvas
        
            Parameters
            ----------
            obj : numpy array
                    image object to be drawn on top of blank
            
            top_left_obj : tuple (row, col)
                    the top left row and column of where you want the obj to be placed
            
            Returns
            ----------
            bool
                wether the placement was succesful or not
        """
        
        bot_right_obj = (top_left_obj[0]+obj.shape[0], top_left_obj[1]+obj.shape[1])
        
        if top_left_obj[0] < self.top_left[0] and top_left_obj[1] < self.top_left[1]:
            return False
        
        if bot_right_obj[0] > self.bot_right[0] and bot_right_obj[1] > self.bot_right[1]:
            return False
        
        min_row = top_left_obj[0]
        max_row = top_left_obj[0] + obj.shape[0]
        
        min_col = top_left_obj[1]
        max_col = top_left_obj[1] + obj.shape[1]
        
        if np.max(self.can_place[min_row:max_row, min_col:max_col] == 1):
            return False

        self.can_place[min_row:max_row, min_col:max_col]  = 1
        self.img[min_row:max_row, min_col:max_col] = obj
    
        return True
        
                                  
        
        