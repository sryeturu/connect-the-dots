import numpy as np


def post_process(yolo_output, obj_thresh, nms_thresh):

    yolo_output = objectness_threshold(yolo_output, obj_thresh)
    yolo_output = transform_bbox_values(yolo_output)
    yolo_output = non_max_supression(yolo_output, nms_thresh)

    return yolo_output


def objectness_threshold(yolo_output, thresh):
    mask = np.where(yolo_output[0,:,4] > thresh)
    result = yolo_output[0, mask[0], :]

    return result


def transform_bbox_values(yolo_output):

    result = yolo_output.copy()

    result[:,0] = (yolo_output[:,0] - yolo_output[:,2]/2.0)
    result[:,1] = (yolo_output[:,1] - yolo_output[:,3]/2.0)
    result[:,2] = (yolo_output[:,0] + yolo_output[:,2]/2.0)
    result[:,3] = (yolo_output[:,1] + yolo_output[:,3]/2.0)

    return result


def iou(a, b):
    
    x1_a = a[0]
    y1_a = a[1]
    x2_a = a[2]
    y2_a = a[3]
    
    x1_b = b[0]
    y1_b = b[1]
    x2_b = b[2]
    y2_b = b[3]
    
    area_of_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_of_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    # finding intersection square
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)
    
    if (x2_inter - x1_inter) < 0 or (y2_inter - y1_inter) < 0:
        return 0
    else:
        area_of_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    intersection_over_union = area_of_inter / (area_of_a + area_of_b - area_of_inter)
    return intersection_over_union


def non_max_supression(yolo_output, nms_remove_threshold):
    
    clean_detections = []
    
    num_classes = yolo_output.shape[1] - 5
    pred_classes = np.argmax(yolo_output[:, 5:], axis=1) 
    
    
    for cur_class_idx in range(num_classes):
    
        cur_boxes = np.where(pred_classes == cur_class_idx)[0]    

        if len(cur_boxes) == 0:
            continue     

        cur_boxes = yolo_output[cur_boxes]   
        cur_boxes[(-1*cur_boxes[:,4]).argsort()]
        removed = set()

        for j in range(len(cur_boxes)):
            if j in removed:
                continue
                
            clean_detections.append((cur_boxes[j][:4], cur_class_idx, cur_boxes[j][4]))
            
            for k in range(j+1, len(cur_boxes)):
                if iou(cur_boxes[j], cur_boxes[k]) > nms_remove_threshold:
                    removed.add(k)
    
    return clean_detections
