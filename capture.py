import cv2 as cv
import numpy as np

from image_utils import adaptive_thresh

# some help here from : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    
    if event == cv.EVENT_LBUTTONUP:
        points.append((x,y))
    
    if len(points) == 2:
        cv.rectangle(frame, points[0], points[1], (0), 2)
        cv.imshow('capture', adaptive_thresh(frame))

        key = cv.waitKey(0)
        
        if key == 99: # 'c' key
            np.save('image_capture', frame)
            print('saved screenshot to image_capture.npy')

        points = []

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame, (1280, 720)) 
    cv.imshow('capture', adaptive_thresh(frame))
    cv.setMouseCallback('capture', mouse_callback)

    key = cv.waitKey(15)

    if key == 27: # exit on ESC
        break

cap.release()
