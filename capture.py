import cv2 as cv
import numpy as np
from constants import NUM_DIRECTORIES

from image_utils import adaptive_thresh, get_number_of_images
from blank import write_to_cfg

class Capture:
    
    # some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

    def __init__(self):
        self.points = []
        
        self.blanks = []
        self.nums = []
        self.dots = []
        self.backgrounds = []

        self.frame = None
        self.window_name = 'capture'
        self.freeze = 0

        
    def mouse_callback(self, event, x, y, flags, param):
        
        if event == cv.EVENT_LBUTTONUP:
            self.points.append((x,y))
        else:
            return
        
        if len(self.points) == 2:
            
            cur_frame = self.frame.copy()
            min_col,min_row = self.points[0]  # opencv garbage
            max_col, max_row = self.points[1]
            img = cur_frame[min_row:max_row, min_col:max_col]
            
            cv.rectangle(self.frame, self.points[0], self.points[1], (0), 2)
            cv.imshow('capture', self.frame)
            
            print("want to save this image? b(blank) n(num) d(dot) o(background) q(no)")
            key = cv.waitKey(0)

            if key == 98: # blank
                self.blanks.append((cur_frame, (min_row,min_col), (max_row,max_col)))
                print('saved blank')
            elif key == 110: # number
                print('ok, what number?')
                num = int(input())
                self.nums.append((img, num))
                print('saved number')
            elif key == 100: # dot
                self.dots.append(img)
                print('saved dot')
            elif key == 111: # backgrounds
                self.backgrounds.append(img)
                print('saved other')
                
            self.freeze = 0 # unfreeze if frozen
            self.points = [] # reset points

            
    def save_captures(self):
        
        for img, num in self.nums:
            num_cnt = get_number_of_images('nums/' + NUM_DIRECTORIES[num])
            
            np.save('nums/' + NUM_DIRECTORIES[num] + '/' + str(num_cnt + 1), img)  # fix hard coded path slash?
            
        cur_dot_idx = get_number_of_images('dots/') + 1
        for img in self.dots:
            np.save('dots/' + str(cur_dot_idx), img)  
            cur_dot_idx += 1
        
        cur_background_idx = get_number_of_images('background/') + 1
        for img in self.others:
            np.save('background/' + str(cur_background_idx), img)  
            cur_other_idx += 1
        
        cur_blank_idx = get_number_of_images('blanks/') + 1
        blanks_to_config = {}
        for img, top_left, bot_right in self.blanks:
            np.save('blanks/' + str(cur_blank_idx), img)  
            blanks_to_config[cur_blank_idx] = (top_left, bot_right)
            cur_blank_idx += 1
        
        if len(blanks_to_config) > 0:
            write_to_cfg(blanks_to_config)

    def run(self):

        cap = cv.VideoCapture(0)
                
        while True:
            
            if self.freeze == 0:
                _, self.frame = cap.read()
                
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                self.frame = cv.resize(self.frame, (1280, 720)) 
                self.frame = adaptive_thresh(self.frame)           

            cv.imshow('capture', self.frame)
            cv.setMouseCallback('capture', self.mouse_callback)


            key = cv.waitKey(15)

            if key == 27: # exit on ESC
                break
            elif key == 102:
                self.freeze = 400 # freeze for 400 frames
            elif key == 117:
                self.freeze = 0
                
        cap.release()
        
        self.save_captures()
            
Capture().run()