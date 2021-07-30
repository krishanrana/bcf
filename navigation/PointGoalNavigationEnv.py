#! /usr/bin/env python3
# Laser Based PointGoalNavigation Gym Environment
# Author: Jake Bruce, Krishan Rana

import numpy as np, cv2, sys, os, math, time
from gym import spaces
from Box2D import *
import os

PATH = os.path.dirname(os.path.realpath(__file__))

#========================================================
# HELPERS

def rnd(mn,mx): return np.random.random()*(mx-mn)+mn

#--------------------------------------------------------

class RaycastCallback(b2RayCastCallback):
    def __init__(self, **kwargs): super(RaycastCallback, self).__init__(**kwargs); self.hit = False; self.fixture = None; self.points  = []; self.normals = []
    def ReportFixture(self, fixture, point, normal, fraction): self.hit = True; self.fixture = fixture; self.points.append(point); self.normals.append(normal); return 1.0

#========================================================
# ENV CLASS

class PointGoalNavigation:

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs):
        self.__dict__.update(dict(
            w = 2000, h = 1000, xmn = -6, xmx = 6, ymn = -3, ymx = 3,
            wait = 1, timeout = 500, k = 0.1, t = 0.0005, eps = 0.025,
            angle_min = -0.5*np.pi, angle_max=0.5*np.pi,
            laser_range = 1.5, num_beams = 180, laser_noise = 0.01, velocity_max=1, omega_max=1, env_type=4, fix_locs=False, reward_type="sparse"))
        self.__dict__.update(kwargs)

        self.bg_img = cv2.resize(cv2.cvtColor(np.random.randint(225,256,(self.h//8,self.w//8)).astype(np.uint8), cv2.COLOR_GRAY2BGR), dsize=(self.w,self.h), interpolation=cv2.INTER_NEAREST)       

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)
        self.reward_range = [-1000, 1000]
        self.num_laser_samples = self.num_beams
        self.num_bins = 15
        self.observation_space = spaces.Box(low=-1, high=1, shape=[self.num_bins+4])
        angle_span  = self.angle_max - self.angle_min
        angle_span += angle_span / self.num_laser_samples
        self.laser_angles = [self.angle_min+i/self.num_laser_samples*angle_span for i in range(self.num_laser_samples)]
        self._max_episode_steps = self.timeout
        self.collided = False
        self.goal_radius = 0.4
        self.pixels_per_meter = 500
        self.laser_obs = np.zeros(self.num_laser_samples)
        self.obs_loc = [-5, -5]
        self.actions_prev = [0, 0]
        self.done = False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

    def w2p(self,x,y): return (int((x-self.xmn)/(self.xmx-self.xmn)*self.w), int(self.h-(y-self.ymn)/(self.ymx-self.ymn)*self.h))
    def w2r(self,r)  : return  int(r/(self.xmx-self.xmn)*self.w)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

    def reset(self):
        self.world = b2World(gravity=(0,0))

        # outer walls
        wall_top      = self.world.CreateStaticBody(position=( 0, self.ymx), shapes=b2PolygonShape(box=(self.xmx,0.05)))
        wall_bottom   = self.world.CreateStaticBody(position=( 0,self.ymn), shapes=b2PolygonShape(box=(self.xmx,0.05)))
        wall_right    = self.world.CreateStaticBody(position=( self.xmx, 0), shapes=b2PolygonShape(box=(0.05,self.xmx)))
        wall_left     = self.world.CreateStaticBody(position=(self.xmn, 0), shapes=b2PolygonShape(box=(0.05,self.xmx)))
        self.outer_walls = [wall_top, wall_bottom, wall_right, wall_left]

        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 0: Bug Trap
        if self.env_type == 0:
            #barrier_top  = self.world.CreateStaticBody(position=(0, 0.3), shapes=b2PolygonShape(box=(0.3, 0.025)))
            #barrier_bottom = self.world.CreateStaticBody(position=(0, -0.3), shapes=b2PolygonShape(box=(0.3, 0.025)))
            #barrier_mid   = self.world.CreateStaticBody(position=(     0.3, 0.0), shapes=b2PolygonShape(box=(0.025, 0.3)))
            #self.barrier_walls = [barrier_top, barrier_bottom, barrier_mid]
            self.barrier_walls = []
        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 1: Randomly changing boxes
        if self.env_type == 1:
            self.barrier_block1 = self.world.CreateStaticBody(position=(0.5*3, rnd(-0.8, 0.8)*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block2 = self.world.CreateStaticBody(position=(0.0*3, rnd(-0.8, 0.8)*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block3 = self.world.CreateStaticBody(position=(-0.5*3, rnd(-0.8, 0.8)*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block4 = self.world.CreateStaticBody(position=(-1.0*3, rnd(-0.8, 0.8)*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block5 = self.world.CreateStaticBody(position=(-1.5*3, rnd(-0.8, 0.8)*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5]
        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 2: Simple
        if self.env_type == 2:
            self.barrier_block1 = self.world.CreateStaticBody(position=( 0.8*3, 0*3), shapes=b2PolygonShape(box=(0.08*3,0.2*3)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.8*3, 0*3), shapes=b2PolygonShape(box=(0.08*3,0.2*3)))
            self.barrier_block3 = self.world.CreateStaticBody(position=( 0*3, 0.7*3), shapes=b2PolygonShape(box=(0.2*3,0.25*3)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( 0*3, -0.7*3), shapes=b2PolygonShape(box=(0.2*3,0.25*3)))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4]
        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 3: Complex
        if self.env_type == 3:
            self.barrier_block1 = self.world.CreateStaticBody(position=( -1.4*3, 0.4*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -1.0*3, -0.1*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_block3 = self.world.CreateStaticBody(position=( 0.1*3, 0.2*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            #self.barrier_block4 = self.world.CreateStaticBody(position=( -0.1*3, 0.5*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            #self.barrier_block5 = self.world.CreateStaticBody(position=( -0.8*3, -0.6*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08, 0.1*3))))
            #self.barrier_block6 = self.world.CreateStaticBody(position=( 0.1*3, -0.2*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_block7 = self.world.CreateStaticBody(position=( 1.1*3, -0.1*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_block8 = self.world.CreateStaticBody(position=( 0.5*3, -0.7*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            #self.barrier_block9 = self.world.CreateStaticBody(position=( 1.2*3, 0.8*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_block10 = self.world.CreateStaticBody(position=( -0.2*3, -0.8*3), shapes=b2PolygonShape(box=(rnd(0.1*3, 0.2*3),rnd(0.08*3, 0.1*3))))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3,  self.barrier_block7, self.barrier_block8,self.barrier_block10]
        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 4: Complex 2
        if self.env_type == 4:
            self.barrier_block1 = self.world.CreateStaticBody(position=( 0.8*3, 0.6*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.8*3, 0.6*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            #self.barrier_block3 = self.world.CreateStaticBody(position=( 0.0*3, 0.2*3), shapes=b2PolygonShape(box=(0.25*3,0.07*3)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( 0.0, 0.0), shapes=b2PolygonShape(box=(0.25*3,0.17*3)))
            self.barrier_block5 = self.world.CreateStaticBody(position=( -1.3*3, 0*3), shapes=b2PolygonShape(box=(0.07*3,0.25*3)))
            self.barrier_block6 = self.world.CreateStaticBody(position=( 1.3*3, 0*3), shapes=b2PolygonShape(box=(0.07*3,0.25*3)))
            self.barrier_block7 = self.world.CreateStaticBody(position=( -0.8*3, -0.6*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block8 = self.world.CreateStaticBody(position=( 0.8*3, -0.6*3), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block9 = self.world.CreateStaticBody(position=( 0*3, 0.8*3), shapes=b2PolygonShape(box=(0.25*3,0.07*3)))
            self.barrier_block10 = self.world.CreateStaticBody(position=( 0*3, -0.8*3), shapes=b2PolygonShape(box=(0.25*3,0.07*3)))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block4, self.barrier_block5, 
                                self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]

        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 5: Complex 3

        if self.env_type == 5:
            self.barrier_block1 = self.world.CreateStaticBody(position=( 0, 0), shapes=b2PolygonShape(box=(0.1*3,0.1*3)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.5*3, 0.7*3), shapes=b2PolygonShape(box=(0.1*3,0.15*3)))
            self.barrier_block3 = self.world.CreateStaticBody(position=( -1.2*3, 0.0*3), shapes=b2PolygonShape(box=(0.2*3,0.05*3)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( -1.35*3, 0.25*3), shapes=b2PolygonShape(box=(0.05*3,0.2*3)))
            self.barrier_block5 = self.world.CreateStaticBody(position=( -1.2*3, -0.5*3), shapes=b2PolygonShape(box=(0.08*3,0.15*3)))
            self.barrier_block6 = self.world.CreateStaticBody(position=( -0.1*3, -0.8*3), shapes=b2PolygonShape(box=(0.2*3,0.05*3)))
            self.barrier_block7 = self.world.CreateStaticBody(position=( 0.15*3, -0.65*3), shapes=b2PolygonShape(box=(0.05*3,0.2*3)))
            self.barrier_block8 = self.world.CreateStaticBody(position=( 1.2*3, 0.5*3), shapes=b2PolygonShape(box=(0.4*3,0.05*3)))
            self.barrier_block9 = self.world.CreateStaticBody(position=( 0.9*3, -0.5*3), shapes=b2PolygonShape(box=(0.2*3,0.3*3)))
            self.barrier_block10 = self.world.CreateStaticBody(position=( 0.3*3, 0.8*3), shapes=b2PolygonShape(box=(0.05*3,0.2*3)))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5, 
                                self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]

        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 6: Complex 6

        if self.env_type == 6:
            self.barrier_block1 = self.world.CreateStaticBody(position=( 0.8*3, 0.6*3), shapes=b2CircleShape(radius=(0.2*2)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.8*3, 0.6*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            self.barrier_block3 = self.world.CreateStaticBody(position=( -0.2*3, 0.2*3), shapes=b2CircleShape(radius=(0.2*2)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( 0.2*3, -0.2*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            self.barrier_block5 = self.world.CreateStaticBody(position=( -1.3*3, 0*3), shapes=b2CircleShape(radius=(0.2*2)))
            self.barrier_block6 = self.world.CreateStaticBody(position=( 1.3*3, 0*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            self.barrier_block7 = self.world.CreateStaticBody(position=( -0.8*3, -0.6*3), shapes=b2CircleShape(radius=(0.2*2)))
            self.barrier_block8 = self.world.CreateStaticBody(position=( 0.8*3, -0.6*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            self.barrier_block9 = self.world.CreateStaticBody(position=( 0*3, 0.8*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            self.barrier_block10 = self.world.CreateStaticBody(position=( 0*3, -0.8*3), shapes=b2CircleShape(radius=(0.2*2)))
            self.circular_blocks = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5, 
                                self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]
            self.barrier_walls = []

        #-----------------------------------------------------------------------------------------------------------------------------------------------#

        if self.env_type == 7:
            #self.barrier_block1 = self.world.CreateStaticBody(position=( 0.8*3, 0.6*3), shapes=b2CircleShape(radius=(0.2*2)))
            #self.barrier_block2 = self.world.CreateStaticBody(position=( -0.8*3, 0.6*3), shapes=b2CircleShape(radius=(0.3*1.5)))
            #self.barrier_block3 = self.world.CreateStaticBody(position=( -0.2*3, 0.1*3), shapes=b2CircleShape(radius=(0.2*2)))
            #self.barrier_block4 = self.world.CreateStaticBody(position=( 0.7*3, -0.2*3), shapes=b2CircleShape(radius=(0.2*1.5)))
            #self.barrier_block5 = self.world.CreateStaticBody(position=( -1.3*3, 0*3), shapes=b2CircleShape(radius=(0.2*2)))

            self.barrier_block1 = self.world.CreateStaticBody(position=( 0.8*3, 0.6*3), shapes=b2PolygonShape(box=(0.2*1.5,0.2*1.5)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.8*3, 0.6*3), shapes=b2PolygonShape(box=(0.3*1.5,0.3*1.5)))
            self.barrier_block3 = self.world.CreateStaticBody(position=( -0.2*3, 0.1*3), shapes=b2PolygonShape(box=(0.2*1.5,0.2*1.5)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( 0.7*3, -0.2*3), shapes=b2PolygonShape(box=(0.2*1.5,0.2*1.5)))
            self.barrier_block5 = self.world.CreateStaticBody(position=( -1.3*3, 0*3), shapes=b2PolygonShape(box=(0.2*1.5,0.2*1.5)))
            
            self.barrier_block6 = self.world.CreateStaticBody(position=( -0.1*3, -0.8*3), shapes=b2PolygonShape(box=(0.2*3,0.05*3)))
            self.barrier_block7 = self.world.CreateStaticBody(position=( 0.15*3, -0.65*3), shapes=b2PolygonShape(box=(0.05*3,0.2*3)))
            self.barrier_block8 = self.world.CreateStaticBody(position=( -1.2*3, -0.8*3), shapes=b2PolygonShape(box=(0.4*3,0.5*3)))
            self.barrier_block9 = self.world.CreateStaticBody(position=( 1.5*3, -0.5*3), shapes=b2PolygonShape(box=(0.2*3,0.2*3)))
            self.barrier_block10 = self.world.CreateStaticBody(position=( 0.3*3, 0.8*3), shapes=b2PolygonShape(box=(0.05*3,0.2*3)))

            #self.circular_blocks = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5]
            self.circular_blocks = []
            #self.barrier_walls = [self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5,
                                    self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]



        #-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 5: Empty World
        if self.env_type == 8:
            self.barrier_walls = []
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------#

        obstacle_mask = np.zeros([self.h, self.w])
        
        for wall in self.barrier_walls:
            points = []
            for local_point in wall.fixtures[0].shape.vertices:
                world_point = wall.GetWorldPoint(local_point)
                pix_point = self.w2p(*world_point)
                points.append(pix_point)
            pt1 = np.array(points[3])
            pt2 = np.array(points[1])
            # Dilate with radius of robot
            obstacle_mask[pt1[1]-40:pt2[1]+40, pt1[0]-40:pt2[0]+40] = 1

        # Draw the mask
        #obstacle_mask = obstacle_mask*255
        #cv2.imshow('wind', obstacle_mask), cv2.waitKey(2500)


        # Search for obstacle free goal location
        while(1):
            goalx = rnd(3.5, 5.0)
            goaly = rnd(-2.5, 2.5)
            goalpt = self.w2p(*np.array([goalx, goaly]))
            if obstacle_mask[goalpt[1], goalpt[0]] == 1:
                continue
            else:
                break
        self.goal = np.array([goalx, goaly])


        # Search for obstacle free robot location
        while(1):
            robotx = rnd(-5.5, -3.5)
            roboty = rnd(-2.5, 2.5)
            robotpt = self.w2p(*np.array([robotx, roboty]))
            if obstacle_mask[robotpt[1], robotpt[0]] == 1:
                continue
            else:
                break
        self.robot_loc = np.array([robotx, roboty])

        if not self.fix_locs:
            swap = np.random.choice([1,2])
            if swap == 1:
                self.goal = np.array([robotx, roboty])
                self.robot_loc = np.array([goalx, goaly])

        # Initialise Agent
        #self.agent_body  = self.world.CreateDynamicBody(position=(-1.3, 0), angle=0, angularVelocity=0, linearDamping=20.0, angularDamping=30.0)
        self.agent_body  = self.world.CreateDynamicBody(position=(self.robot_loc[0], self.robot_loc[1]), angle=rnd(-np.pi,np.pi), angularVelocity=0, linearDamping=20.0, angularDamping=30.0)
        #self.agent_body = self.world.CreateKinematicBody(position=(rnd(-0.9,0.9), rnd(-0.9,-0.8)), angle=rnd(-np.pi,np.pi), angularVelocity=0, linearDamping=20.0, angularDamping=30.0)
        self.agent_shape = self.agent_body.CreateFixture(shape=b2CircleShape(pos=(0,0), radius=0.3), density=0.1, friction=0.3)
        self.agent_body.mass = 5
        self.trail_buffer = []

        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        agent_point = self.w2p(*agent_loc)
        self.trail_buffer.append(agent_point)

        # temp_obstacle_mask = cv2.resize(obstacle_mask,(int(self.w/10), int(self.h/10)))
        # temp_obstacle_mask[int(goalpt[1]/10), int(goalpt[0]/10)] = 5
        # temp_obstacle_mask[int(agent_point[1]/10), int(agent_point[0]/10)] = 9
        # np.savetxt('maptest_eval.txt', temp_obstacle_mask.astype(int), fmt='%d', delimiter='')
        # self.mask = temp_obstacle_mask
        
        self.timestep = 0

        return self._obs()


#-----------------------------------------------------------------------------------------------------------------------------------------------#

    def render(self):

        agent_color = (0,0,0)
        # Make a copy as we dont wan to write over original image
        img = self.bg_img.copy()

        # draw goal
        # radius = self.w2r(self.agent_shape.shape.radius) 
        # cv2.circle(img, self.w2p(*self.goal), int(self.pixels_per_meter * self.goal_radius*0.6), (99,245,66), 45)
        # cv2.circle(img, self.w2p(*self.goal), int(self.pixels_per_meter * self.goal_radius*0.35), (0,0,255), 50)
        # cv2.circle(img, self.w2p(*self.goal), int(self.pixels_per_meter * self.goal_radius*0.20), (255,0,0), -1)


        # draw laser rays
        laser_img = img.copy()
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        points = [self.w2p(*agent_loc)]
        for angle, dist in zip(self.laser_angles, self.laser_obs):
            end = agent_loc + [dist*np.cos(self.agent_body.angle + angle), dist*np.sin(self.agent_body.angle + angle)]
            p2 = self.w2p(*end)
            points.append(p2)
        cv2.fillPoly(laser_img, pts=[np.array(points)], color=(128,0,128))
        cv2.polylines(laser_img, pts=[np.array(points)], isClosed=True, color=(255,0,255), thickness=3)
        for angle, dist in zip(self.laser_angles, self.laser_obs):
            end = agent_loc + [dist*np.cos(self.agent_body.angle + angle), dist*np.sin(self.agent_body.angle + angle)]
            p2 = self.w2p(*end)
            cv2.line(laser_img, points[0], p2, (255,0,255), 3)
        img = laser_img//4 + img//4*3 # blend laser image at a low alpha

        # draw walls
        for wall in self.outer_walls + self.barrier_walls:
            points = []
            for local_point in wall.fixtures[0].shape.vertices:
                world_point = wall.GetWorldPoint(local_point)
                pix_point = self.w2p(*world_point)
                points.append(pix_point)
            cv2.fillConvexPoly(img, points=np.array(points), color=(64,64,64))
            cv2.polylines(img, pts=[np.array(points)], isClosed=True, color=(0,0,0), thickness=8)

        if self.env_type == 6 or self.env_type == 7:
            for ob in self.circular_blocks:
                cv2.circle(img, self.w2p(*ob.position), self.w2r(ob.fixtures[0].shape.radius), (64,64,64), -1)
                cv2.circle(img, self.w2p(*ob.position), self.w2r(ob.fixtures[0].shape.radius), (0,0,0), 10)

        # Agent position and shape data
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        pix_point = self.w2p(*agent_loc)
        radius    = self.w2r(self.agent_shape.shape.radius)

        # Draw agent
        cv2.circle(img, pix_point, radius, agent_color, -1)
       
        # draw orientation vector
        cv2.circle(img, self.w2p(*(np.array(agent_loc) + [0.25*np.cos(self.agent_body.angle), 0.25*np.sin(self.agent_body.angle)])), self.w2r(0.1), (0,255,255), -1)

        # show image
        cv2.namedWindow("PointGoalNavigation", cv2.WINDOW_NORMAL)
        cv2.imshow("PointGoalNavigation", img)
        key = cv2.waitKey(self.wait) & 0xff
        if   key ==       27: sys.exit(0)
        elif key == ord('r'): self.reset()
        elif key == ord(' '):
            key = cv2.waitKey(self.wait)&0xff
            while key != ord(' ') and key != 27: key = cv2.waitKey(self.wait)&0xff
            if key == 27: sys.exit(0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def goal_achieved(self):
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        to_goal = self.goal - np.array(agent_loc)
        return np.linalg.norm(to_goal) < self.goal_radius

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def step(self, action):

        # Scale the actions by their maximums
        #print(action)
        action = np.clip(action, -1, 1) #uncomment if i am using ppo as it does not clip actions
        #print(action)
        lin = float(action[0] * 10)
        omega = float(action[1] * 15)

        velocity = (lin*np.cos(self.agent_body.angle), lin*np.sin(self.agent_body.angle))
        
        self.agent_body.linearVelocity = (velocity)
        self.agent_body.angularVelocity = (omega)

        # Previous range to goal
        prev_dist = self._obs()[-2]
        self.actions_prev = action

        # Simulate
        self.world.Step(1/60, 10, 10)
        self.world.ClearForces()

        dist = self._obs()[-2]

        self.timestep += 1

        # Defining the rewards
        dense_rew = ((prev_dist - dist) * 5000)

        #dense_rew = ((prev_dist - dist) * 150) #- (self.timestep * 0.1)

        # if lin < 0:
        #      dense_rew -= 150

        if self.collided:
             dense_rew -= 50


        

        if self.timestep > self.timeout: 
            self.done = True
            if self.reward_type == "sparse":
                return self._obs(),   0.0, True,  {}
            elif self.reward_type == "dense":
                return self._obs(),   dense_rew, True,  {}

        elif self.goal_achieved(): 
            self.done = True
            if self.reward_type == "sparse": 
                return self._obs(),   1.0, False,  1.0
            elif self.reward_type == "dense":
                return self._obs(),   1.0, False,  1.0
        else: 
            self.done = False
            if self.reward_type == "sparse": 
                return self._obs(),  -0.01, False, {}
            elif self.reward_type == "dense":
                return self._obs(),  dense_rew, False, {}

        #done = 1 if self.timestep > self.timeout else 0
        #reward = 1 if self.goal_achieved() else 0
        #return self._obs(),reward

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def _laser_rays(self):
        agent_loc = np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        return [(agent_loc, agent_loc + [self.laser_range*np.cos(self.agent_body.angle + angle), self.laser_range*np.sin(self.agent_body.angle + angle)]) for angle in self.laser_angles]

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def _get_position_data(self):
        robot_angle = np.arctan2(np.sin(self.agent_body.angle), np.cos(self.agent_body.angle))
        to_goal = self.goal - np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        angle_to_goal = robot_angle - np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        dist_to_goal  = np.linalg.norm(to_goal)
        robot_loc = np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        laser_scan = self.laser_obs

        return dist_to_goal, angle_to_goal, robot_loc, robot_angle, laser_scan, self.goal, self.obs_loc


    def _obs(self):
        # Generate laser scan
        laser_samples = np.zeros(self.num_laser_samples)
        for i,(start, end) in enumerate(self._laser_rays()):
            callback = RaycastCallback()
            #print('start: ', start, ' end: ', end)
            self.world.RayCast(callback, start, end)
            laser_samples[i] = min([np.linalg.norm(start - point) for point in callback.points]) if len(callback.points) > 0 else self.laser_range
        num_hits = (laser_samples < self.laser_range).sum()
        self.laser_obs = laser_samples.copy()
        # Applying noise to laser scan
        self.laser_obs[laser_samples < self.laser_range] *= np.random.normal(1,self.laser_noise,num_hits)
        # Get global data
        dist_to_goal, angle_to_goal, _, _, _, _,_ = self._get_position_data()
        hit = (laser_samples < self.laser_range).astype(np.float)
        self.collided = np.any((laser_samples < 0.1).astype(np.float) == 1.0)

        laser_scan = np.zeros(self.num_bins)
        laser_hit = np.zeros(self.num_bins)
        div_factor = int(self.num_laser_samples/self.num_bins)
        for i in range(self.num_bins):
            laser_scan[i] = np.mean(self.laser_obs[i*div_factor:(i*div_factor+div_factor)]) 
            laser_hit[i] = (laser_scan[i] < self.laser_range).astype(np.float)
            
        return np.concatenate([laser_scan, # distances
                              self.actions_prev,
                              [dist_to_goal], # linear distance to goal
                              [angle_to_goal]]) # angular difference to goal

    
    def seed(self, seed=0):
        np.random.seed(seed)
        self.action_space.np_random.seed(seed)
        return
