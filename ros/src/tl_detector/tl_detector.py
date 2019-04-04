#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
IMAGE_PROCESSING_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.is_site = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.image_count = 0
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.is_site = self.config['is_site']
        self.is_simulator = not self.is_site

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                rospy.loginfo("publish light wp index: {}".format(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                rospy.loginfo("publish light wp index: {}".format(light_wp))
            self.state_count += 1
            rate.sleep()
        

        #rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoints_2d = [[wp.pose.pose.position.x, 
                              wp.pose.pose.position.y] for wp in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        self.image_count += 1
        
        if self.image_count < IMAGE_PROCESSING_COUNT_THRESHOLD:
            return
        
        self.image_count = 0
        
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            rospy.loginfo("publish light wp index: {}".format(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            rospy.loginfo("publish light wp index: {}".format(light_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_wp_indx = None

        if self.waypoint_tree is not None:
            closest_wp_indx = self.waypoint_tree.query([pose.position.x,pose.position.y], 1)[1]
            
        return closest_wp_indx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return light.state
        
        
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        predicted_state = self.light_classifier.get_classification(cv_image)
        rospy.loginfo("Predicted state: {}".format(self.light_to_string(predicted_state)))

        if self.is_simulator:
            rospy.loginfo("Ground truth: {}".format(self.light_to_string(light.state)))

        return predicted_state

    def light_to_string(self, state):
        if state == TrafficLight.RED:
            return "red"
        elif state == TrafficLight.YELLOW:
            return "yellow"
        elif state == TrafficLight.GREEN:
            return "green"
        return "unknown"

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
        else:
            return -1, TrafficLight.UNKNOWN
        
        if car_position is None:
            return -1, TrafficLight.UNKNOWN

        best_index_distance = len(self.waypoints.waypoints)
        closest_light = None
        best_stop_line_index = None

        for i, light in enumerate(self.lights):
            stop_line_pose = Pose()
            stop_line_pose.position.x = stop_line_positions[i][0]
            stop_line_pose.position.y = stop_line_positions[i][1]
            stop_line_index = self.get_closest_waypoint(stop_line_pose)
            
            index_distance = stop_line_index - car_position

            if index_distance >= 0 and index_distance < best_index_distance:
                best_index_distance = index_distance
                closest_light = light
                best_stop_line_index = stop_line_index

        if closest_light:
            state = self.get_light_state(closest_light)
            rospy.loginfo("Closest stop waypoint index: {}, state: {}".format(best_stop_line_index, self.light_to_string(state)))
            return best_stop_line_index-1, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')