import tensorflow as tf
from yolo_config import parse_config
import numpy as np
import cv2
from timeit import default_timer as timer


class Yolo:
    
    
    def __init__(self, weight_file, cfg_file, device='CPU', weight_file_offset=4*4):
        print(weight_file)
        fp = open(weight_file, 'rb')
        fp.seek(weight_file_offset) # skipping any header stuff from weight file
        
        self.device = device
        self.model_cfg, self.op_cfg = parse_config(cfg_file)
        self.weights = np.fromfile(fp, dtype = np.float32)
        
        self.build_graph()
    

    def preprocess(self, img):

        img = cv2.resize(img, (int(self.model_cfg['height']), int(int(self.model_cfg['width']))), interpolation=1)
        img = img[np.newaxis,:,:,:]/255.0
        return img[:, :, : , 0, np.newaxis]


    def predict(self, img):

        start = timer()
        
        img = self.preprocess(img)
        print(img.shape)
                         
        with tf.Session() as sess:
            pred = sess.run(fetches=[self.detection_tensor], feed_dict={self.input_tensor:img})[0]
                    
        print('completed in %f seconds' % (timer() - start))
        
        return pred


    def build_graph(self):
        
        tf.reset_default_graph()

        output_maps = []
        detections = []
                
        self.weights_counter = 0
        
        with tf.device('/device:%s:0' % self.device):
            self.input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, int(self.model_cfg['height']), int(self.model_cfg['width']), int(self.model_cfg['channels'])), name='input_image')
            
            inp = self.input_tensor
            
            for i, op in enumerate(self.op_cfg):
                if op['type'] == 'convolutional':

                    inp = self.add_conv_op(inp, op)

                elif op['type'] == 'shortcut':            
                    inp = output_maps[int(op['from'])] + inp

                elif op['type'] == 'route':            
                    if isinstance(op['layers'], list):
                        a, b = [int(x) for x in op['layers']]
                        inp = tf.concat((output_maps[a],output_maps[b]), axis=3)
                    else:
                        print('route',i+int(op['layers']))
                        inp = output_maps[int(op['layers'])]

                elif op['type'] == 'upsample':          
                    stride = int(op['stride'])

                    inp = tf.image.resize_images(images=inp, size=[inp.shape[1]*stride, inp.shape[2]*stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                elif op['type'] == 'yolo':        
                    print('yolo layer shape: ', inp.shape)
                    detection_map = self.add_yolo_layer(inp, op)

                    detections.append(detection_map)

                else:
                    raise Exception('unkown op type')
                
                output_maps.append(inp)
          
            
            del self.weights_counter
            
            self.detection_tensor = tf.concat(detections, axis=1)
            
            
        
    def add_yolo_layer(self, inp, op):
    
        image_height = int(self.model_cfg['height'])
        image_width = int(self.model_cfg['width'])

        feature_map_height = inp.shape[1].value
        height_shrink = image_height // feature_map_height

        feature_map_width = inp.shape[2].value
        weight_shrink = image_width // feature_map_width
    
        a = [float(x) for x in op['anchors']]
        
        a = np.array([(a[i], a[i+1]) for i in range(0, len(a)-1, 2)], dtype=np.float32)

        anchors = a[[int(i) for i in op['mask']]] #change 

        res = tf.reshape(tensor=inp, shape=(-1, feature_map_height * feature_map_width * len(anchors), int(op['classes']) + 5))

        x_y = tf.sigmoid(res[:,:,:2]) # x and y sigmoids
        h_w = tf.exp(res[:,:,2:4]) # exponentiate height and width
        iou = tf.sigmoid(res[:,:,4])[:,:,tf.newaxis] # iou sigmoids
        class_scores = tf.sigmoid(res[:,:,5:]) # class score sigmoids

        offsets = []
        

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for i in range(len(anchors)):
                    offsets.append([x, y])

        x_y += offsets

        anchors[:, 0] = anchors[:, 0] / height_shrink
        anchors[:, 1] = anchors[:, 1] / weight_shrink

        anchors = np.tile(anchors, (feature_map_height * feature_map_width, 1))
        h_w = h_w * anchors


        h = tf.expand_dims(h_w[:, :, 0] * height_shrink, axis=2)
        w = tf.expand_dims(h_w[:, :, 1] * weight_shrink, axis=2)

        x = tf.expand_dims(x_y[:, :, 0] * weight_shrink, axis=2)
        y = tf.expand_dims(x_y[:, :, 1] * height_shrink, axis=2)

        bbox_attrs = [x, y, h, w, iou, class_scores]

        res = tf.concat(values=bbox_attrs, axis=2) 

        return res

    def add_conv_op(self, input_tensor, op): 

        kernel_size  = int(op['size'])
        strides = int(op['stride'])
        out_channels = int(op['filters'])
        in_channels = input_tensor.shape[-1].value


        if 'pad' in op:               
            padding_amount = (kernel_size - 1) // 2
            input_tensor = tf.pad(tensor=input_tensor, paddings=([0,0], [padding_amount,padding_amount], [padding_amount,padding_amount], [0,0]))   

        ''' weights are loaded witht the bn/bias then conv weights'''

        if 'batch_normalize' in op:
            num_of_weights = out_channels
            bn_beta = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
            self.weights_counter += num_of_weights

            bn_gamma = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
            self.weights_counter += num_of_weights

            bn_mean = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
            self.weights_counter += num_of_weights

            bn_var = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
            self.weights_counter += num_of_weights

        else:
            # bias instead
            num_of_weights = out_channels
            bias = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
            self.weights_counter += num_of_weights


        # conv weights
        weights_shape = [kernel_size, kernel_size, in_channels, out_channels]

        num_of_weights = kernel_size * kernel_size * in_channels * out_channels    
        conv_weights = self.weights[self.weights_counter : self.weights_counter + num_of_weights]
        self.weights_counter += num_of_weights

        conv_weights = conv_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)
        conv_weights = conv_weights.transpose((2,3,1,0))

        ''' done loading any weights/creating variables, now we can add the conv operations to the graph'''

        output_tensor = tf.nn.conv2d(input=input_tensor, filter=conv_weights, strides=(1,strides,strides,1), padding='VALID')

        if 'batch_normalize' in op: 
            mean, var = tf.nn.moments(output_tensor, axes=[0,1,2])
            output_tensor = tf.nn.batch_normalization(x=output_tensor, mean=bn_mean, variance=bn_var, offset=bn_beta, scale=bn_gamma, variance_epsilon=1e-5) 
            #output_tensor = tf.nn.batch_normalization(x=output_tensor, mean=mean, variance=var, offset=bn_beta, scale=bn_gamma, variance_epsilon=1e-5) 

        else:
            output_tensor = output_tensor = tf.nn.bias_add(value=output_tensor, bias=bias, data_format='NHWC')

        if 'activation' in op:  
            if op['activation'] == 'leaky':
                output_tensor = tf.nn.leaky_relu(output_tensor, alpha=.1)
            else:
                pass # activation is linear

        return output_tensor