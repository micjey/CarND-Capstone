from styx_msgs.msg import TrafficLight
import rospy
import os
import cv2
import numpy as np
import tensorflow as tf
import yaml

TRAFFIC_LIGHT_SCORE_THRESHOLD = 0.5
MODELS_DIRECTORY = '/traffic_light_models/'
MODEL_FILENAME_SIM = "frozen_inference_graph_traffic_130.pb"
MODEL_FILENAME_REAL = "frozen_inference_graph_real_merged_130.pb"

class TLClassifier(object):

    def __init__(self):
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config['is_site']
        self.is_simulator = not self.is_site
        
        traffic_light_green = {'name': 'Green'}
        traffic_light_red = {'name': 'Red'}
        traffic_light_yellow = {'name': 'Yellow'}

        self.label_dict = {1: traffic_light_green, 2: traffic_light_red, 3: traffic_light_yellow}
        
        model_path = os.path.dirname(os.path.realpath(__file__))
        model_path += MODELS_DIRECTORY
        
        if self.is_simulator:
            model_path += MODEL_FILENAME_SIM
        else:
            model_path += MODEL_FILENAME_REAL
      
        self.build_model_graph(model_path)
        rospy.loginfo("Classifier initialized")

    def build_model_graph(self, model_path):
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                saved_graph = fid.read()
                od_graph_def.ParseFromString(saved_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.model_graph)

        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.model_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model_graph.get_tensor_by_name('num_detections:0')    
     
    def get_classification(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        
        with self.model_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], 
                                                          feed_dict={self.image_tensor: image_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        predicted_traffic_light = self.detect_traffic_light(boxes, scores, classes, num, image)
        return predicted_traffic_light
    
    def create_feature(self, rgb_image):
        
        #Convert image to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        #Create and return a feature value and/or vector
        brightness_channel = hsv[:,:,2]
        rows = brightness_channel.shape[0]
        cols = brightness_channel.shape[1]
        mid = int(cols/2)

        red_region = brightness_channel[:int(rows/3),(mid-10):(mid+10)]
        yellow_region = brightness_channel[int(rows/3):int(2*rows/3),(mid-10):(mid+10)]
        green_region = brightness_channel[int(2*rows/3):,(mid-10):(mid+10)]

        feature = [0,0,0]
        feature[0] = np.mean(green_region)
        feature[1] = np.mean(red_region)
        feature[2] = np.mean(yellow_region)

        return feature
    
    def predict_traffic_class(self,traffic_img):
        standard_im = cv2.resize(np.copy(traffic_img), (32, 32))
        rgb_image = cv2.cvtColor(standard_im, cv2.COLOR_BGR2RGB)
        max_index = np.argmax(self.create_feature(rgb_image)) + 1
        
        predicted_traffic_light = TrafficLight.UNKNOWN
        
        if max_index == 1:
            predicted_traffic_light = TrafficLight.GREEN
        elif max_index == 2:
            predicted_traffic_light = TrafficLight.RED
        elif max_index == 3:
            predicted_traffic_light = TrafficLight.YELLOW

        return predicted_traffic_light
    
    def detect_traffic_light(self, boxes, scores, classes, num, bgr_img):
        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im_height, im_width, channels = image.shape
     
        predicted_traffic_light = TrafficLight.UNKNOWN
        opt_idx = scores.argmax()
        
        if scores[opt_idx] > TRAFFIC_LIGHT_SCORE_THRESHOLD:
            ymin = int(boxes[0,opt_idx,0]*im_height)
            xmin = int(boxes[0,opt_idx,1]*im_width)
            ymax = int(boxes[0,opt_idx,2]*im_height)
            xmax = int(boxes[0,opt_idx,3]*im_width)
       
            traffic_img = image[ymin:ymax,xmin:xmax]
            predicted_traffic_light = self.predict_traffic_class(traffic_img)
            
        return predicted_traffic_light
    