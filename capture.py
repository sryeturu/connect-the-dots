import cv2 as cv
import numpy as np

from image_utils import adaptive_thresh


class Capture:
    
    # some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

    def __init__(self):
        self.points = []
        
        self.blanks = []
        self.nums = []
        self.dots = []
        
        self.frame = None

        
    def mouse_callback(self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONUP:
            self.points.append((x,y))

        if len(self.points) == 2:
            cv.rectangle(self.frame, self.points[0], self.points[1], (0), 2)
            cv.imshow('capture', adaptive_thresh(self.frame))
            
            print("want to save this image? esc(no), 'b'(blank), 'n'(num), 'd'(dot)")
            key = cv.waitKey(0)
            
            x1,y1 = self.points[0]
            x2, y2 = self.points[1]
            img = self.frame[x1:x2, y1:y2]
            
            if key == 98: # blank
                self.blanks.append((img, self.points[0], self.points[1]))
            elif key == 110: # number
                print('ok, what number?')
                num = int(input())
                key = cv.waitKey(0)
                
                print('saved it')
            elif key == 100:
                print('dot')
            else:
                pass
                
            
            
        
            self.points = []
        


    def run(self):

        cap = cv.VideoCapture(0)
        

        while True:
            _, self.frame = cap.read()
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.frame = cv.resize(self.frame, (1280, 720)) 
            cv.imshow('capture', adaptive_thresh(self.frame))
            cv.setMouseCallback('capture',self.mouse_callback)


            key = cv.waitKey(15)

            if key == 27: # exit on ESC
                break

        cap.release()

            
Capture().run()