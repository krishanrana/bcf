import numpy as np
import cv2
import time
#from matplotlib import pyplot as plt
import scipy.ndimage
from collections import deque

        
# All angles in this method have to be in robot frame and not global frame
# This should be more reactive than the force method

class P_controller():
    def __init__(self, env):
        self.fov = 360
        self.Kv = 0.1
        self.Kw = 0.8
        self.env = env

    def clipAngle(self, angle):
        #clip angle between 0 and FOV
        if angle < 0:
            angle = angle + self.fov
        elif angle > self.fov:
            angle = angle - self.fov
        return int(angle)

    def attractive_field(self, angle_to_goal):
        mapTo360 = angle_to_goal + np.pi
        #print(np.rad2deg(angle_to_goal))
        #init map with range of FOV 0 - FOV
        goal_bearing = self.clipAngle((np.rad2deg(mapTo360)))
        attraction_field = np.zeros([(self.fov+1)]) # 0 - FOV inculsive

        #set value of map to one at location of goal
        attraction_field[int(goal_bearing)] = 1

        #gradient is how sharp the attraction in the map is
        gradient = 1/(self.fov/2)


        for angle in range(int(self.fov/2)):

            loc = int(self.clipAngle(goal_bearing - angle))
            attraction_field[loc] = 1 - angle * gradient

            loc = int(self.clipAngle(goal_bearing + angle))
            attraction_field[loc] = 1 - angle * gradient

        
        return attraction_field

    def compute_action(self):

        dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = self.env._get_position_data()
        Kw = 2# 2
        Kv = 0.08 # 0.1
        att = self.attractive_field(angle_to_goal)
        result = att
        peak = max(result)
        index = np.where(result==peak)[0][0]
        heading = np.deg2rad(index - self.fov/2)        
        fov_map = np.arange(-self.fov/2, self.fov/2+1)

        omega = -heading * self.Kw
        
        vel = (10 * self.Kv) * (1.0 - min(0.8 * abs(omega), 0.95)) # 10 instaead of distt-goal

        omega = np.clip(omega, -1, 1)
        vel = np.clip(vel, -1, 1)

        return np.array([vel, omega])


class PotentialFieldsController():

    def __init__(self, env):
        self.fov = 360
        #self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True)
        self.buffer = deque(maxlen=2)
        self.buffer.append(0)
        self.buffer.append(0)
        self.env = env

    def clipAngle(self, angle):
        #clip angle between 0 and FOV
        if angle < 0:
            angle = angle + self.fov
        elif angle > self.fov:
            angle = angle - self.fov

        return angle

    def attractive_field(self, angle_to_goal):
        mapTo360 = angle_to_goal + np.pi
        #print(np.rad2deg(angle_to_goal))
        #init map with range of FOV 0 - FOV
        goal_bearing = self.clipAngle((np.rad2deg(mapTo360)))
        attraction_field = np.zeros([(self.fov+1)]) # 0 - FOV inculsive

        #set value of map to one at location of goal
        attraction_field[int(goal_bearing)] = 1

        #gradient is how sharp the attraction in the map is
        gradient = 1/(self.fov/2)

        #iterate through each angle in the fov map and compute linear relation to goal angle
        #ie compute ramp profile of map

        for angle in range(int(self.fov/2)):

            loc = int(self.clipAngle(goal_bearing - angle))
            attraction_field[loc] = 1 - angle * gradient

            loc = int(self.clipAngle(goal_bearing + angle))
            attraction_field[loc] = 1 - angle * gradient

        # fov_map = np.arange(-self.fov/2, self.fov/2+1)
        # plt.plot(fov_map, attraction_field)
        # plt.show(block=False)
        # plt.pause(0.00000001)
        # plt.cla()

        return attraction_field


    def repulsive_field(self, laser_scan):
        hit = np.flip((laser_scan < 1.5))
        struct = scipy.ndimage.generate_binary_structure(1, 1)
        hit = scipy.ndimage.binary_dilation(hit, structure=struct, iterations=35).astype(hit.dtype) #30
        #hit = 1 - laser_scan*2
        repulsive_field = np.zeros([(self.fov+1)])
        repulsive_field[int(self.fov/4) : int(3*self.fov/4)] = hit
        #repulsive_field[int(self.fov/8) : int(7*self.fov/8)] = hit
        return repulsive_field


    def compute_action(self):

        dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = self.env._get_position_data()

        #print('ls: ', laser_scan)

        Kw = 2# 2
        Kv = 0.08 # 0.1
        att = self.attractive_field(angle_to_goal)
        rep = self.repulsive_field(laser_scan)

        result = att - rep
        peak = max(result)
        index = np.where(result==peak)[0][0]
        heading = np.deg2rad(index - self.fov/2)
        self.buffer.append(heading)
        if abs(self.buffer[0] - self.buffer[1]) > 10:
            heading = self.buffer[1]
            print('here')
        
        fov_map = np.arange(-self.fov/2, self.fov/2+1)

        # Compute a repulsive angular velocity to ensure robot steers away from obstacle
        #rep_angle = self.fov/2 - np.where(laser_scan == np.min(laser_scan))[0][0]
        omega = -heading * Kw
        
        vel = (10 * Kv) * (1.0 - min(0.8 * abs(omega), 0.95)) 

        omega = np.clip(omega, -1, 1)
        vel = np.clip(vel, -1, 1)

        return np.array([vel, omega])


