import numpy as np
import glob
import os
import cv2 as cv

from scipy import ndimage

from config import parse_cfg
from image_utils import get_number_of_images, adaptive_thresh, get_corners, get_scaled_position

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
            x1, y1 = canvases[key][0]
            x2, y2 = canvases[key][1]
       
            f.writelines('\n[%d]' % key)
            f.writelines('\ntop_left =  %d, %d' % (x1, y1))
            f.writelines('\nbot_right =  %d, %d' % (x2, y2))
            f.writelines('\n[end]\n')
            
            
class Canvas:
    
    def __init__(self, img, top_left, bot_right):
        
        self.img = img
        
        # specify these are paper attributes
        self.top_left = top_left
        self.bot_right = bot_right        
        self.bot_left = top_left[0], bot_right[1]
        self.top_right = bot_right[0], top_left[1]    
        
        self.contours = np.array([self.top_left, self.top_right, self.bot_right, self.bot_left])
        
        
    def draw_on_background(self, obj, top_left_obj):
        
        if top_left_obj[1] + obj.shape[0] >= self.img.shape[0] or top_left_obj[0] + obj.shape[1] >= self.img.shape[1]:
            return False

        top_left_obj, top_right_obj, bot_right_obj, bot_left_obj = get_corners(top_left_obj, obj)
                
        x1, y1 = top_left_obj #top left
        x2, y2 = bot_right_obj #bot right        
        
        mask = np.zeros_like(self.img)
        mask = cv.fillPoly(mask, [self.contours], color=1) # 0/1 mask containg paper polygon region
        
        img2 = np.copy(self.img)
        img2[y1:y2, x1:x2] = obj # draw on copy
        img2 = img2 * (1-mask) # zero out paper polygon region
        
        self.img = (mask*self.img) + img2
      
                
    def rotate(self, degrees):
        rad = np.deg2rad(degrees)
        rt = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        
        original_shape = self.img.shape
        self.img = adaptive_thresh(ndimage.rotate(self.img, angle=degrees, cval=255))
        
        nw = np.dot(rt, self.contours.T - np.array(original_shape)[[1,0]].reshape(2,-1)/2.0) + np.array([self.img.shape[1]/2, self.img.shape[0]/2]).reshape(2,-1)
        nw = nw.astype(np.int32)
        
        self.contours = nw.T
        
        self.top_left = tuple(self.contours[0, :])
        self.top_right = tuple(self.contours[1, :])
        self.bot_right = tuple(self.contours[2, :])
        self.bot_left = tuple(self.contours[3, :])

    def resize(self, new_size):
        
        img_size = self.img.shape[::-1]
        
        self.top_left = get_scaled_position(self.top_left, img_size, new_size)
        self.top_right = get_scaled_position(self.top_right, img_size, new_size)
        self.bot_right = get_scaled_position(self.bot_right, img_size, new_size)
        self.bot_left = get_scaled_position(self.bot_left, img_size, new_size)
        
        self.contours = np.array([self.top_left, self.top_right, self.bot_right, self.bot_left])
        
        self.img = cv.resize(self.img, new_size)
        self.img = adaptive_thresh(self.img)

    def all_corners_inside_paper(self, top_left_obj, obj):

        for corner in get_corners(top_left_obj, obj):
             if cv.pointPolygonTest(self.contours, (corner[1], corner[0]), False) < 0.0:
                    return False
    
        return True

    def draw_on_paper(self, obj, top_left_obj, can_overlap=False):
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

        if top_left_obj[1] + obj.shape[0] >= self.img.shape[0] or top_left_obj[0] + obj.shape[1] >= self.img.shape[1]:
            return False

        obj_corners = get_corners(top_left_obj, obj)
        for corner in obj_corners:
            if cv.pointPolygonTest(self.contours, corner , False) < 0.0: # corner x,y
                return False
        
        x1, y1 = obj_corners[0] #top left
        x2, y2 = obj_corners[2] #bot right

        overlay = np.copy(obj)
        
        if can_overlap or np.alltrue(self.img[y1:y2, x1:x2] == 255) :
            self.img[y1:y2, x1:x2]  = overlay
            return True

        for row in range(obj.shape[0]):
            for col in range(obj.shape[1]):
                canvas_pixle = self.img[y1+row, x1+col]
                obj_pixle = obj[row, col]
                
                if canvas_pixle == 0 and  obj_pixle == 0:
                    return False
                
                if canvas_pixle == 0 and obj_pixle == 255:
                    overlay[row, col] = 0
                elif canvas_pixle == 255 and obj_pixle == 0:
                    overlay[row, col] = 0
                elif canvas_pixle == 255 and obj_pixle == 255:
                    overlay[row, col] = 255
                else:
                    raise ValueError('pixel color mismatch')
        
        self.img[y1:y2, x1:x2] = overlay
        
        return True
        
                                     
        