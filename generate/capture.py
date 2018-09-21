import cv2 as cv
import numpy as np

from image_utils import get_number_of_images, adaptive_thresh
from canvas import write_to_cfg
from config import parse_cfg


class Capture:
    
    # some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

    def __init__(self, cfg):
        
        self.cfg = cfg
        img_size = int(cfg['size']['width']), int(cfg['size']['height'])
        self.img_size = img_size
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
            
            x1, y1 = self.points[0] 
            x2, y2 = self.points[1]
            region_crop = cur_frame.copy()[y1:y2, x1:x2]
            
            display = cur_frame.copy()
            cv.rectangle(display, self.points[0], self.points[1], (0), 1)
            cv.imshow('capture', display)
            
            print("save an image: c(canvas) n(num) d(dot) b(background) r(drawing)")
            key = cv.waitKey(0)

            if key == 99: # canvas
                self.canvases.append((cur_frame, (x1, y1), (x2, y2)))
                self.freeze = False # probably want to unfreeze (if frozen) wehenever we get a canvas pic

                print('saved canvas')                
            elif key == 110: # number
                print('ok, what number?')
                num = int(input())
                self.nums.append((region_crop, num))
                print('saved number')
            elif key == 100: # dot
                self.dots.append(region_crop)
                print('saved dot')
            elif key == 98: # backgrounds
                self.backgrounds.append(region_crop)
                print('saved background')
            elif key == 114: # drawings
                self.drawings.append(region_crop)
                print('saved drawing')
                
            self.points = [] # reset points

            
    def save_captures(self):                
        
        cv.imwrite('result.png', self.frame)
        
        cur_canvas_idx = get_number_of_images('canvases/') + 1
        canvas_to_config = {}
        for img, top_left, bot_right in self.canvases:
            cv.imwrite('canvases/' + str(cur_canvas_idx) + '.png', img)  
            canvas_to_config[cur_canvas_idx] = (top_left, bot_right)
            cur_canvas_idx += 1
        
        if len(canvas_to_config) > 0:
            write_to_cfg(canvas_to_config)
            
        for img, num in self.nums:
            num_cnt = get_number_of_images('nums/' + str(num))
            
            cv.imwrite('nums/' + str(num) + '/' + str(num_cnt + 1) + '.png', img)  # fix hard coded path slash?
            
        cur_dot_idx = get_number_of_images('dots/') + 1
        for img in self.dots:
            cv.imwrite('dots/' + str(cur_dot_idx) + '.png', img)  
            cur_dot_idx += 1
        
        cur_background_idx = get_number_of_images('backgrounds/') + 1
        for img in self.backgrounds:
            cv.imwrite('backgrounds/' + str(cur_background_idx) + '.png', img)  
            cur_background_idx += 1
            
        cur_drawing_idx = get_number_of_images('drawings/') + 1
        for img in self.drawings:
            cv.imwrite('drawings/' + str(cur_drawing_idx) + '.png', img)  
            cur_drawing_idx += 1
        

    def run(self):

        cap = cv.VideoCapture(0)
        
        print('(s) to save a picture (a) toggle adaptive thresh (f) toggle freeze')
        
        ad_thresh = False
        
        while True:
            
            if self.freeze == False:
                _, self.frame = cap.read()
            
                if ad_thresh:
                    
                    self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
     
                    if len(self.cfg['adaptive_thresh']) == 2:
                        block_size = int(cfg['adaptive_thresh']['block_size'])
                        constant = int(cfg['adaptive_thresh']['contant'])
                        self.frame = adaptive_thresh(self.frame, block_size, constant)
                    else:
                        self.frame = adaptive_thresh(self.frame)

            cv.imshow('capture', self.frame)
            cv.setMouseCallback('capture', self.mouse_callback)
            
            key = cv.waitKey(15)
      
            if key == 27: # exit on ESC
                break
            elif key == 102: # pressed 'f'
                self.freeze = not self.freeze
            elif key == 97: # press 'a'
                ad_thresh = not ad_thresh
            elif key == 115:
                cv.imwrite('result.png', self.frame)
                print('saved image as result.png')
                
        cap.release()
        
        self.save_captures()

        
if __name__ == '__main__':
    
    cfg = parse_cfg('capture.cfg')
    Capture(cfg).run()