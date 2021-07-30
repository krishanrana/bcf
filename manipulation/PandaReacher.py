
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
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation

        self.ep_step += 1
        
        # Generate reward
        reward = self.gen_reward_sparse()

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
            q[i] = random.uniform(self.qlim[0, i] + off, self.qlim[1, i] - off)

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