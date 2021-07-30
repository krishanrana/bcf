#!/usr/bin/env python

import numpy as np
from spatialmath import SE3, SO3
import pdb
import roboticstoolbox as rb

class RRMC_controller():

    def __init__(self, env):
        self.env = env
        self.panda = rb.models.DH.Panda()

    def p_servo(self, wTe, wTep, gain=2):
        '''
        Position-based servoing.

        Returns the end-effector velocity which will cause the robot to approach
        the desired pose.

        :param wTe: The current pose of the end-effecor in the base frame.
        :type wTe: SE3
        :param wTep: The desired pose of the end-effecor in the base frame.
        :type wTep: SE3
        :param gain: The gain for the controller
        :type gain: float
        :param threshold: The threshold or tolerance of the final error between
            the robot's pose and desired pose
        :type threshold: float

        :returns v: The velocity of the end-effecotr which will casue the robot
            to approach wTep
        :rtype v: ndarray(6)
        :returns arrived: True if the robot is within the threshold of the final
            pose
        :rtype arrived: bool

        '''

        if not isinstance(wTe, SE3):
            wTe = SE3(wTe)

        if not isinstance(wTep, SE3):
            wTep = SE3(wTep)


        # Pose difference
        eTep = wTe.inv() * wTep

        # Translational velocity error
        ev = eTep.t

        # Angular velocity error
        ew = eTep.rpy() * np.pi/180

        # Form error vector
        e = np.r_[ev, ew]
        
        dist = np.array(self.env.agent.get_tip().get_position()) - np.array(self.env.target.get_position())

        # Desired end-effector velocity
        v = gain * e
        
        return v
    
    def fkine(self):
        # Tip pose in world coordinate frame
        pose = SE3(self.env.agent.get_tip().get_position())*SE3.Eul(self.env.agent.get_tip().get_orientation())
        return pose
    
    def target_pose(self):
        # Target pose in world coordinate frame
        pose = SE3(self.env.target.get_position())*SE3.Eul(self.env.target.get_orientation())
        return pose


    def compute_action(self, gain=1):

        try:
            self.panda.q = self.env.agent.get_joint_positions()
            v = self.p_servo(self.panda.fkine(self.panda.q), self.target_pose(), gain=0.3)
            v[3:] *= 10
            action = np.linalg.pinv(self.panda.jacobe(self.panda.q)) @ v

        except np.linalg.LinAlgError:
            action = np.zeros(self.env.n)
            print('Fail')

        return action
