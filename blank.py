
class Blank:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        sef.top_left = top_left
        self.bot_right = bot_right
        
        self.min_row = top_left[0]
        self.max_row = bot_right[0]
                                 
        self.min_col = top_left[1]
        self.max_col = bot_right[1]
        
        self.can_place = np.zeros(shape=img.shape)
    
            
    def place_object(obj, top_left_obj):
        
        bot_right_obj = (top_left_obj[0]+obj.shape[0], top_left_obj[1]+obj.shape[1])
        
        if top_left_obj[0] < self.top_left[0] and top_left_obj[1] < self.top_left[1]:
            return false
        
        if bot_right_obj[0] > self.bot_right[0] and bot_right_obj[1] > self.bot_right[1]:
            return false
        
        row1 = top_left_obj[0]
        row2 = top_left_obj[0] + img.shape[0]
        
        col1 = top_left_obj[1]
        col2 = top_left_obj[1] + img.shape[1]
        
        if np.max(self.can_place[row1:row2, col1:col2] == 1):
            return false

        self.can_place[row1:row2, col1:col2] == 1
        self.img[row1:row2, col1:col2] == obj
        
        return true
        
                                  
        
        