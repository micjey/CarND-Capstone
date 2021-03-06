#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32 , Float32
from scipy.spatial import KDTree
from std_msgs.msg import String
import numpy as np
import math


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
STOP_LINE_COUNT_BEFORE = 5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.base_lane = None
        self.pose = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.track_waypoint_count = -1
        self.stopline_wp_idx = -1
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.base_waypoints_subscriber = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_cooord = self.waypoints_2d[closest_idx - 1]

        #Equation for hyperplane through CLosest Co-ordinates
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_cooord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self):
        lane = Lane()
        closest_index = self.get_closest_waypoint_idx()
        last_index = closest_index + LOOKAHEAD_WPS
        
        if last_index < self.track_waypoint_count:
            base_waypoints = self.base_lane.waypoints[closest_index:last_index]
        else:
            index_offset = last_index - self.track_waypoint_count
            last_index = self.track_waypoint_count - 2
            base_waypoints = self.base_lane.waypoints[closest_index:last_index]
            base_waypoints += self.base_lane.waypoints[0:index_offset]
        
        if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= last_index:
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_index)
        
        self.final_waypoints_pub.publish(lane)
 
    def decelerate_waypoints(self, waypoints, closest_index):
        result = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_index = max(self.stopline_wp_idx - closest_index - STOP_LINE_COUNT_BEFORE, 0)
            dist = self.distance(waypoints, i, stop_index)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            result.append(p)
            
        return result

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_lane = waypoints
        self.track_waypoint_count = len(waypoints.waypoints)
        
        if self.waypoints_2d == None:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
           
    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data    
    pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')