from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType, PrimitiveShape, TextureMappingMode
import numpy as np
from gym import spaces
import math
import roboticstoolbox as rp 
import pdb
import random
from spatialmath import SE3

SCENE_FILE = join(dirname(abspath(__file__)), 'scene_reinforcement_learning_env_custom.ttt')
POS_MIN, POS_MAX = [0.2, -0.7, 0.1], [0.6, 0.7, 0.7]
ANG_MIN, ANG_MAX = [50, 50, 5], [120,120,60]


class ReacherEnv(object):

    def __init__(self,**kwargs):
        self.headless = True if 'headless' not in kwargs or kwargs['headless'] == 1 else False
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=self.headless)
        self.pr.start()
        self.agent = Panda()
        self.panda = rp.models.DH.Panda()
        self.panda_utils = rp.models.DH.Panda()
        # self.gripper = PandaGripper()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.action_limits = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
        self.observation_space = spaces.Box(low=-1, high=1, shape=[20])
        self.nSubtasks = 1
        self.eps_since_reset = 0
        self._max_episode_steps = kwargs['max_ep_steps']
        self.n = 7
        self.qlim = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                            [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]])
        
        self.rang = self.qlim[1, :] - self.qlim[0, :]
        self.reward_dense = True if 'dense' not in kwargs or kwargs['dense'] == 1 else False 


    def _reinit(self):
        self.pr.shutdown()
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=self.headless)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.eps_since_reset = 0

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        ePos = self.agent_ee_tip.get_position()
        tPos = self.target.get_position()
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               ePos,
                               tPos-ePos])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        #pos = list(np.random.uniform(POS_MIN, POS_MAX))

        # Training set goals
        # goals = [SE3(np.array([[ 0.84073806, -0.44629211, -0.30656624,  0.24963363],
        #         [-0.53174833, -0.57392387, -0.62278014, -0.68898124],
        #         [ 0.10199618,  0.68661106, -0.71983473,  0.44332631],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.10110619, -0.10436278,  0.98938665,  0.47462406],
        #         [-0.9572867 , -0.28098526,  0.0681869 , -0.16202907],
        #         [ 0.27088689, -0.9540208 , -0.12831445,  0.80934663],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.05411526, -0.78143632,  0.62163399,  0.10838407],
        #         [-0.56873205,  0.48757787,  0.66242861,  0.88864022],
        #         [-0.82074076, -0.38939067, -0.41804248,  0.50389342],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.84245165,  0.45252368,  0.29239961, -0.05737615],
        #         [ 0.51652456, -0.52402569, -0.67719972,  0.21326123],
        #         [-0.153224  ,  0.7215396 , -0.67520591,  0.16504184],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.70751599,  0.22229853,  0.67082374,  0.57837032],
        #         [ 0.63700084, -0.61165997, -0.46915031,  0.19818329],
        #         [ 0.3060246 ,  0.75924663, -0.57436356,  0.6470301 ],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.15235587,  0.26497509, -0.95214279,  0.26712485],
        #         [ 0.90608942,  0.34729487,  0.24163658,  0.54208645],
        #         [ 0.39470198, -0.89954126, -0.18717871,  0.54609549],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[ 0.72795795, -0.24767288,  0.63932415,  0.17450122],
        #         [ 0.40465092,  0.90795042, -0.10901225,  0.12978233],
        #         [-0.55347525,  0.33805944,  0.76117078,  1.24661989],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[-0.03822127, -0.6053562 ,  0.79503648,  0.45318418],
        #         [ 0.05406903, -0.79570515, -0.60326598, -0.26448151],
        #         [ 0.99780543,  0.01992926,  0.06314389,  0.83183064],
        #         [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False), 
        # SE3(np.array([[-9.77519945e-01,  2.09534910e-01, -2.34494890e-02,
        #             3.19499706e-01],
        #         [-8.05274322e-04, -1.14927268e-01, -9.93373583e-01,
        #         -3.27127880e-01],
        #         [-2.10841430e-01, -9.71023607e-01,  1.12512429e-01,
        #             2.54358977e-01],
        #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             1.00000000e+00]]), check=False)]

        
        #self.targ = SE3(0,0,0)


        #self.targ = self._find_pose()    
        #self.targ = goals[i]
        #self.pos = self.targ.t
        #self.rot = self.targ.eul('deg')
        self.pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.rot = list(np.deg2rad(np.random.uniform(ANG_MIN, ANG_MAX)))
    
        self.target.set_position(self.pos)
        self.target.set_orientation(self.rot)

        self.agent.set_joint_positions(self.initial_joint_positions)

        self.ep_step = 0
        self.eps_since_reset += 1
        if self.eps_since_reset >= 100:
            self._reinit()
        return self._get_state()

    def gen_reward_dense(self):
        # Reward is negative distance to target
        dist = self.get_dist()
        reward = math.exp(-3*dist)
        return reward
    
    def gen_reward_sparse(self):
    
        dist = self.get_dist()
        robot_pose = self.agent.get_joint_positions()
        reward_manip = self.panda.manipulability(robot_pose)
        if dist < 0.03:
            reward = 1 + (reward_manip * 10)
        else:
            reward = reward_manip
        return reward

    def get_dist(self):
        ePos = self.agent_ee_tip.get_position()
        tPos = self.target.get_position()
        # Reward is negative distance to target
        dist = np.linalg.norm(ePos-tPos)
        return dist

    def step(self, action):
        # print(action)
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        # self.gripper.actuate(action[-1])
        self.pr.step()  # Step the physics simulation

        self.ep_step += 1
        
        # Generate reward
        reward = self.gen_reward_sparse()
        #reward = self.gen_reward_sparse()

        #input('c')

        # print(reward)
        done = True if (self.ep_step >= self._max_episode_steps) else False

        info = {"reward0":reward,
                "overallReward":reward,
                "subtask":0,
                "success":1 if reward > 0.95 else 0}

        return self._get_state(), reward, done, info

    def _rand_q(self, k=0.2):
        q = np.zeros(self.n)
        for i in range(self.n):
            off = k * self.rang[i]
            #print('off: ', off)
            q[i] = random.uniform(self.qlim[0, i] + off, self.qlim[1, i] - off)
            #print('qi', q[i])
            #print(self.qlim[0, i] + off)
            #print(self.qlim[1, i] - off)
        return q
        

    def _find_pose(self):
        q = self._rand_q()
        return self.panda_utils.fkine(q)

    def _check_limit(self):
        off = 0.2

        robot_pose = self.agent.get_joint_positions()


        for i in range(7):
            if robot_pose[i] <= (self.qlim[0, i] + off):
                print('Joint limit hit')
                return True
            
            elif robot_pose[i] >= (self.qlim[1, i] - off):
                print('Joint limit hit')
                return True

        return False
    
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


    def seed(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.np_random.seed(seed)
        return


    
class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


EPISODES = 50
EPISODE_LENGTH = 200

if __name__ == '__main__':
    env = ReacherEnv(headless=False, max_ep_steps=1000)
    agent = Agent()
    replay_buffer = []

    for e in range(EPISODES):

        print('Starting episode %d' % e)
        state = env.reset()
        for i in range(EPISODE_LENGTH):
            action = agent.act(state)
            
            reward, next_state,_,_ = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            state = next_state
            agent.learn(replay_buffer)

    print('Done!')
    env.shutdown()