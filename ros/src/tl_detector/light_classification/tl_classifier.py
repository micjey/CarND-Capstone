from styx_msgs.msg import TrafficLight
import rospy
import os
import numpy as np
import tensorflow as tf
import yaml

TRAFFIC_LIGHT_SCORE_THRESHOLD = 0.5
MODELS_DIRECTORY = '/traffic_light_models/'
MODEL_FILENAME_SIM = "test_frozen_inference_graph_sim.pb"
MODEL_FILENAME_REAL = "test_frozen_inference_graph_sim.pb"

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
        predicted_traffic_light = self.detect_traffic_light(scores, scores.argmax(), classes)
    
        return predicted_traffic_light
    
    def detect_traffic_light(self, scores, highest_score_idx, classes):
        predicted_traffic_light = TrafficLight.UNKNOWN
        
        if scores[highest_score_idx] > TRAFFIC_LIGHT_SCORE_THRESHOLD:
            rospy.logwarn("Current traffic light: {}".format(self.label_dict[classes[highest_score_idx]]['name']))
            if classes[highest_score_idx] == 1:
                predicted_traffic_light = TrafficLight.GREEN
            elif classes[highest_score_idx] == 2:
                predicted_traffic_light = TrafficLight.RED
            elif classes[highest_score_idx] == 3:
                predicted_traffic_light = TrafficLight.YELLOW
        else:
            rospy.logwarn("Could not find any traffic light.")
            
        return predicted_traffic_light