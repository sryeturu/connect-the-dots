import argparse
import cv2 as cv
import numpy as np

from model import Yolo
from post_processing import post_process



def get_nums_and_dots(detection):
    """ returns the nums and dots with the highest
         class scores.

        Parameters
        ----------
        detection : ndarray
        
            output of potential Yolo predictions after obj and nms threhsold.
            has shape of (batch_size * n * (classes + 5))
            where n is the number of
            bounding boxes depending on the model
            
        Returns
        -------
        nums, dots
            tuple containing nums and dots.
            nums is a dict key=num_value, value=(num_bbox, num_value, num_class_score)
            dots is list of items=(dot_bbox, 0, dot_class_score)

        Examples
        --------
        >>> get_nums_and_dots(yolo.predict(img))
        

        ( {1: (array([286.0641 , 460.0583 , 300.0575 , 482.92767], dtype=float32),
           1,
           0.53679866),
          2: (array([255.96031, 350.24774, 269.17938, 374.11603], dtype=float32),
           2,
           0.19776492),
          3: (array([269.4448 , 103.55083, 280.56943, 125.34567], dtype=float32),
           3),
            [(array([280.62265,  99.33054, 292.8276 , 118.1025 ], dtype=float32),
           0,
           0.98406017),
          (array([353.3219  ,  99.18658 , 365.84174 , 118.449394], dtype=float32),
           0,
           0.90541035)] )
    """
    
    
    dots = []
    nums = {}

    for detection in clean_detections:

        bbox = detections[0]
        obj_type = detection[1]
        class_score = detection[2]

        if obj_type == 0:
            dots.append(detection)
        elif obj_type not in nums:
            nums[obj_type] = detection
        elif class_score > nums[obj_type][2]:
            nums[obj_type] = detection
            
            
    return nums, dots




def draw_lines(img, nums, dots, model_inp_size):
    """ draws lines connecting the dots on the passed on the image
        and returns the result.

        Parameters
        ----------
        img : ndarray
            numpy array representing image, should be same image passed into the model
        
        nums :  dict 
            returned dict from get_nums_and_dots function containing nums to draw
        
        dots :  list 
            returned list from get_nums_and_dots function containing dots to draw  
            
        model_inp_size : tuple
            tuple containing (width, height) of the model input. This is needed
            to ensure lines are scaled are drawn correctly scaled

        Returns
        -------
        res : ndarray
            res is the image with the lines drawn on it.

    """   
    dot_centers = []
    removed_dots = set()
    
    get_center = lambda bbox: ((bbox[0] + bbox[2])/2.0, (bbox[1] + bbox[3])/2.0)
    distance = lambda point1, point2: np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    for cur_num in sorted(nums):
        num_center = get_center(nums[cur_num][0])
        
        best_dist = 10000000000
        best_index = -1
        best_center = None
        
        for i, dot_vals in enumerate(dots):
            if i in removed_dots:
                continue
                
            dot_center = get_center(dot_vals[0])
            
            dist = distance(num_center, dot_center) 

            if dist < best_dist:
                best_dist = dist
                best_index = i
                best_center = dot_center
        

        if best_index > -1:
            removed_dots.add(best_index)
            dot_centers.append(best_center)
    
    res = cv.resize(img, model_inp_size)

    for i in range(len(dot_centers)-1):
        pt1 = (int(dot_centers[i][0]), int(dot_centers[i][1]))
        pt2 = (int(dot_centers[i+1][0]), int(dot_centers[i+1][1]))
        
        cv.line(res, pt1, pt2,(0,0,0), 2)

    return res


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-img', required=True)
    parser.add_argument('-cfg', required=True)
    parser.add_argument('-weights', required=True)
    parser.add_argument('-out', default='result')
    parser.add_argument('-obj_thresh', default=.01)
    parser.add_argument('-nms_thresh', default=.2)

    args = parser.parse_args()

    sys_args = {}
    
    sys_args['img_path'] = args.img
    sys_args['cfg_path'] = args.cfg
    sys_args['weights_path'] = args.weights
    sys_args['out_path'] = args.out
    sys_args['obj_thresh'] = args.obj_thresh
    sys_args['nms_thresh'] = args.nms_thresh

    return sys_args


if __name__ == '__main__':
    
    sys_args = parse_args()
    
    yolo = Yolo(sys_args['weights_path'], sys_args['cfg_path'])
    
    img = cv.imread(sys_args['img_path'])
    
    detections = yolo.predict(img)
    clean_detections = post_process(yolo_output=detections, obj_thresh=sys_args['obj_thresh'], nms_thresh=sys_args['nms_thresh'])
    
    nums, dots = get_nums_and_dots(clean_detections)
    result = draw_lines(img, nums, dots, yolo.get_inp_size(return_type='xy'))
    
    OUT_SIZE = 848, 480 # need to add to argparse
     
    result = cv.resize(result, OUT_SIZE)
    
    cv.imwrite(sys_args['out_path'] + '.png', result)
    print('saved to %s.png' % sys_args['out_path'])
    