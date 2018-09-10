import cv2 as cv
import numpy as np
import random
import shutil
import copy

from image_utils import *
from canvas import *
from scipy import ndimage
from sampling import Sampler
from config import parse_cfg


NUM_OF_SAMPLES = 10
MAX_NUMBER = 4

MAX_NUMBERS_TO_DRAW = 20
MAX_DRAWINGS_TO_DRAW = 3
MAX_BACKGROUNDS_TO_DRAW = 5

DRAW_BBOX = False
IMAGE_SIZE = (672, 512) # (width, height)
IMAGE_SIZE = (704, 416) # (width, height)

RESULTS_DIR = 'results'

OBJ_IMAGES_DIR = RESULTS_DIR + '/objs'
OBJ_DATA_FILE = RESULTS_DIR + '/obj.data'
OBJ_NAME_FILE = RESULTS_DIR + '/obj.names'
TRAIN_FILE = RESULTS_DIR + '/train.txt'


def get_above_coords(canvas, dot, num_coords):

    top_left_row = num_coords['min_col'] + random.randint(-20, 20)
    top_left_col = num_coords['min_row'] - dot.shape[0] - random.randint(0, 10)

    return top_left_row, top_left_col 

def get_below_coords(canvas, num, dot_coords):
        
    top_left_col = dot_coords['max_row'] + random.randint(0, 10)
    top_left_row = dot_coords['min_col'] + random.randint(-20, 20)
    
    return top_left_row, top_left_col

def get_right_coords(canvas, num, dot_coords):
    
    top_left_col = dot_coords['min_row'] + random.randint(-20, 20)
    top_left_row = dot_coords['max_col'] + random.randint(0, 10)
    
    return top_left_row, top_left_col

def get_left_coords(canvas, num, dot_coords):
    
    top_left_col = dot_coords['min_row'] + random.randint(-20, 20)
    top_left_row = dot_coords['min_col'] - num.shape[1] - random.randint(0, 10)
    
    return top_left_row, top_left_col

def get_potential_pos(canvas):
    x1 = random.randint(canvas.top_left[0],  canvas.top_right[0])
    y1 = random.randint(canvas.top_left[1],  canvas.bot_left[1])
    
    return x1, y1

def get_yolo_bbox(corners, canvas):
    ''' corners is a tuple of  (top_left, top_right, bot_right, bot_left)'''
    
    x1, y1 = corners[0] #top left
    x2, y2 = corners[2] #bot right
    
    x_mid = (x1 + x2)/2
    y_mid = (y1 + y2) /2
    
    width = x2 - x1
    height = y2 - y1
    
    x_abs = x_mid / canvas.img.shape[1]
    y_abs = y_mid / canvas.img.shape[0]
    
    width_abs = width / canvas.img.shape[1]
    height_abs = height / canvas.img.shape[0]
    
    return x_abs, y_abs, width_abs, height_abs

def write_obj_names(): 
    with open(OBJ_NAME_FILE, 'w') as f:
        f.write('dot\n')
        for i in range(1, MAX_NUMBER+1):      
            f.write(str(i) + '\n')

def write_train_file():
    with open(TRAIN_FILE, 'w') as f:
        for i in range(1, NUM_OF_SAMPLES+1):      
            f.write('%s/img%d.png\n' % (OBJ_IMAGES_DIR, i))

def write_data_file():
    with open(OBJ_DATA_FILE, 'w') as f:
        f.write('classes = %d\n' % (MAX_NUMBER + 1))
        f.write('train = %s\n' % TRAIN_FILE)
        f.write('valid = %s\n' % 'results/test.txt')
        f.write('names = %s\n' % OBJ_NAME_FILE )
        f.write('backup = backup/')


def main():
    
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(RESULTS_DIR)
    os.makedirs(OBJ_IMAGES_DIR)

    func = [get_above_coords, get_below_coords, get_left_coords, get_right_coords]

    canvas_sampler = Sampler(get_canvases('canvases'))
    dot_sampler = Sampler(get_img_data('dots'))
    background_sampler = Sampler(get_img_data('backgrounds'))
    drawing_sampler = Sampler(get_img_data('drawings'))
    
    num_sampler = [None]
    
    for i in range(1, MAX_NUMBER+1):
        imgs = get_img_data('nums/' + str(i))
        num_sampler.append(Sampler(imgs))
    
    imgs = []
    bbox_strings = []
    
    for img_idx in range(NUM_OF_SAMPLES):
        
        if img_idx > 0 and img_idx % 25 == 0:
            print('completed generating %d images' % img_idx)
            
        canvas = copy.deepcopy(canvas_sampler.get_sample())
        
        bboxs = {}

        # drawing drawings
        num_of_drawings = random.randint(0, MAX_DRAWINGS_TO_DRAW)
        for i in range(num_of_drawings):
            drawing = drawing_sampler.get_sample()
            drawing = random_resize(drawing, 1.0, 1.0)
            x1, y1 = get_potential_pos(canvas)

            while not canvas.draw_on_paper(drawing, (x1, y1), True):
                x1, y1 = get_potential_pos(canvas)

        # drawing numbers
        num_of_nums = random.randint(0, MAX_NUMBERS_TO_DRAW)
        for i in range(num_of_nums):
            cur_num = random.randint(1, MAX_NUMBER)
            num = num_sampler[cur_num].get_sample()
            num = random_resize(num, .9, 0.9)
            num = adaptive_thresh(ndimage.rotate(num, angle=random.randint(-15, 15), cval=255))

            x1, y1 = get_potential_pos(canvas)
            
            while not canvas.draw_on_paper(num, (x1, y1)):
                x1, y1 = get_potential_pos(canvas)

            top_left, top_right, bot_right, bot_left = get_corners((x1, y1), num)

            if cur_num not in bboxs:
                bboxs[cur_num] = []

            bboxs[cur_num].append((top_left, top_right, bot_right, bot_left))

            num_coords = {}
            num_coords['min_row'] = top_left[1]
            num_coords['max_row'] = bot_right[1]
            num_coords['min_col'] = top_left[0]
            num_coords['max_col'] = bot_right[0]

            np.random.shuffle(func)

            scalar = .8   
            dot = dot_sampler.get_sample()
            dot = cv.resize(dot, ( int(dot.shape[0]*scalar), int(dot.shape[1]*scalar)))
            dot = adaptive_thresh(dot)

            for f in func:
                top_left = f(canvas, dot, num_coords)
                if canvas.draw_on_paper(dot, top_left):
                    if 0 not in bboxs:
                        bboxs[0] = []

                    bboxs[0].append(get_corners(top_left, dot))
                    break
                    
        deg = 0 # rotations not working out so well
        
        org_center = np.array(canvas.img.shape[::-1]).reshape(2,-1) / 2

        canvas.rotate(deg)
        rot_center = np.array(canvas.img.shape[::-1]).reshape(2,-1) / 2
        rot_size = canvas.img.shape[::-1]
    
        background_count = random.randint(0, MAX_BACKGROUNDS_TO_DRAW)
        for i in range(background_count):

            background = background_sampler.get_sample()
            background = random_resize(background, .8, 1.3)

            y1 = random.randint(0, canvas.img.shape[0])
            x1 = random.randint(0, canvas.img.shape[1])

            while canvas.all_corners_inside_paper((x1, y1), background):
                y1 = random.randint(0, canvas.img.shape[0])
                x1 = random.randint(0, canvas.img.shape[1])

            canvas.draw_on_background(background, (x1, y1))
        
        if deg != 0:
            canvas.resize(IMAGE_SIZE)
        
        rad = np.deg2rad(deg)
        rt = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        
        bbox_str = ''
        for k,v in bboxs.items():
            for obj in v:
                rt_bbox = np.dot(rt, np.array(obj).T - org_center) + rot_center
                
                corners = tuple([get_scaled_position(corner, rot_size, IMAGE_SIZE) for corner in [rt_bbox[:, 0], rt_bbox[:, 1], rt_bbox[:, 2], rt_bbox[:, 3]]])
        
                bbox_str += ('%d %f %f %f %f\n' % (k, *get_yolo_bbox(corners, canvas)))
                
                if DRAW_BBOX:            
                    canvas.img = cv.rectangle(canvas.img, corners[0], corners[2], (0), 1)
        

        file_path = OBJ_IMAGES_DIR + '/img' + str(img_idx + 1) 
        
        # writing .png
        cv.imwrite(file_path + '.png', canvas.img)
        # writing .txt
        with open(file_path + '.txt', 'w') as f:
            f.write(bbox_str)
            
    write_obj_names()
    write_train_file()
    write_data_file()
    
if __name__ == '__main__':
    main()