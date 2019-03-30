from pid import PID  
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, wheel_base, wheel_radius, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit, tor_limit):

        self.vehicle_mass = vehicle_mass + GAS_DENSITY * fuel_capacity
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.tor_limit = tor_limit
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        
        self.yaw_controller  = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
       
        self.tau = 0.5
        self.ts = 0.02
        self.vel_lpf = LowPassFilter(self.tau, self.ts)

        self.kp = 0.3
        self.ki = 0.1
        self.kd  = 0.022
        self.min  = 0.0
        self.max = 0.2
        

        self.throttle_controller = PID(self.kp , self.ki, self.kd , self.min , self.max)
        self.last_time = rospy.get_time()


    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):

        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0,0.0,0.0

        current_velocity = self.vel_lpf.filt(current_velocity)

        steering  = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        vel_error  = linear_velocity - current_velocity
        
        current_time  = rospy.get_time()
        sample_time  = current_time  - self.last_time
        self.last_time = current_time

        throttle  = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0

        if linear_velocity == 0.0 and current_velocity < 0.1:
            throttle = 0.0
            brake = self.tor_limit #  NM this is mim torque required for Carla Car at idle

        elif throttle < 0.1 and vel_error < 0.0:
            throttle = 0.0
            decel = max (vel_error , self.decel_limit)
            brake  = abs(decel)* self.vehicle_mass * self.wheel_radius

        return throttle , brake , steering
