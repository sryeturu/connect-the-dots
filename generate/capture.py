import cv2 as cv
import numpy as np

from image_utils import adaptive_thresh, get_number_of_images
from canvas import write_to_cfg

IMAGE_SIZE = (704, 416) # (width, height)

class Capture:
    
    # some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

    def __init__(self):
        self.points = []
        
        self.canvases = []
        self.nums = []
        self.dots = []
        self.backgrounds = []
        self.drawings = []

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
            x1, y1 = self.points[0]  # opencv garbage
            x2, y2 = self.points[1]
            img = cur_frame[y1:y2, x1:x2]
            
            cv.rectangle(self.frame, self.points[0], self.points[1], (0), 1)
            cv.imshow('capture', self.frame)
            
            print("save an image: c(canvas) n(num) d(dot) b(background) r(drawing) u(unfreeze)")
            key = cv.waitKey(0)

            if key == 99: # canvas
                self.canvases.append((cur_frame, (x1, y1), (x2, y2)))
                print('saved blank')                
            elif key == 110: # number
                print('ok, what number?')
                num = int(input())
                self.nums.append((img, num))
                print('saved number')
            elif key == 100: # dot
                self.dots.append(img)
                print('saved dot')
            elif key == 98: # backgrounds
                self.backgrounds.append(img)
                print('saved background')
            elif key == 114: # drawings
                self.drawings.append(img)
                print('saved drawing')
            else:
                self.freeze = 0 # unfreeze
                
            self.points = [] # reset points

            
    def save_captures(self):
        
        cur_canvas_idx = get_number_of_images('canvases/') + 1
        canvas_to_config = {}
        for img, top_left, bot_right in self.canvases:
            np.save('canvases/' + str(cur_canvas_idx), img)  
            canvas_to_config[cur_canvas_idx] = (top_left, bot_right)
            cur_canvas_idx += 1
        
        if len(canvas_to_config) > 0:
            write_to_cfg(canvas_to_config)
            
        for img, num in self.nums:
            num_cnt = get_number_of_images('nums/' + str(num))
            
            np.save('nums/' + str(num) + '/' + str(num_cnt + 1), img)  # fix hard coded path slash?
            
        cur_dot_idx = get_number_of_images('dots/') + 1
        for img in self.dots:
            np.save('dots/' + str(cur_dot_idx), img)  
            cur_dot_idx += 1
        
        cur_background_idx = get_number_of_images('backgrounds/') + 1
        for img in self.backgrounds:
            np.save('backgrounds/' + str(cur_background_idx), img)  
            cur_background_idx += 1
            
        cur_drawing_idx = get_number_of_images('drawings/') + 1
        for img in self.drawings:
            np.save('drawings/' + str(cur_drawing_idx), img)  
            cur_drawing_idx += 1
        

    def run(self):

        cap = cv.VideoCapture(0)
                
        while True:
            
            if self.freeze == 0:
                _, self.frame = cap.read()
                
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                self.frame = cv.resize(self.frame, IMAGE_SIZE) 
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