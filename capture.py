import cv2 as cv
import numpy as np
from constants import NUM_DIRECTORIES

from image_utils import adaptive_thresh, get_number_of_images


class Capture:
    
    # some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

    def __init__(self):
        self.points = []
        
        self.blanks = []
        self.nums = []
        self.dots = []
        
        self.frame = None
        self.window_name = 'capture'
        
        
    def mouse_callback(self, event, x, y, flags, param):
        
        if event == cv.EVENT_LBUTTONUP:
            self.points.append((x,y))
        else:
            return
        
        if len(self.points) == 2:
            cv.rectangle(self.frame, self.points[0], self.points[1], (0), 2)
            cv.imshow('capture', adaptive_thresh(self.frame))
            
            print("want to save this image? 'p'(no), 'b'(blank), 'n'(num), 'd'(dot)")
            key = cv.waitKey(0)
            
            y1,x1 = self.points[0]  # opencv garbage
            y2, x2 = self.points[1]
            img = self.frame[x1:x2, y1:y2]
            
            if key == 98: # blank
                self.blanks.append((img, self.points[0], self.points[1]))
                print('saved blank')
            elif key == 110: # number
                print('ok, what number?')
                num = int(input())
                self.nums.append((img, num))
                print('saved number')
            elif key == 100:
                self.blanks.append((img))
                print('saved dot')
            else:
                pass

            self.points = []
        
    def save_captures(self):
        
        for i in self.nums:
            img, num = i
            imag_count = get_number_of_images('nums/' + NUM_DIRECTORIES[num])
            
            np.save('nums/' + NUM_DIRECTORIES[num] + '/' + str(imag_count + 1), img)  # fix hard coded path slash
            
            
            
    def run(self):

        cap = cv.VideoCapture(0)

        while True:
            _, self.frame = cap.read()
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.frame = cv.resize(self.frame, (1280, 720)) 
            cv.imshow('capture', adaptive_thresh(self.frame))
            cv.setMouseCallback('capture', self.mouse_callback)


            key = cv.waitKey(15)

            if key == 27: # exit on ESC
                break
        
        cap.release()
        
        self.save_captures()
            
Capture().run()