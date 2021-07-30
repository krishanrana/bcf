""" 
Author: Krishan Rana, SpinningUp
Project: Guided Policy Optimisation (GPO) 

"""

import numpy as np
import torch
from torch.optim import Adam
#from torch.utils.tensorboard import SummaryWriter
#from torch.distributions import Normal
from torch.distributions.normal import Normal
import gym
import time
import sac_core as core
from copy import deepcopy
import itertools
import math
import pdb
import wandb
import collections
import sys, os
import random
import ray



HYPERS = collections.OrderedDict()
def arg(tag, default):
    HYPERS[tag] = type(default)((sys.argv[sys.argv.index(tag)+1])) if tag in sys.argv else default
    return HYPERS[tag]


PROJECT_LOG = arg("--project_log", "GPO_2021")

ray.init()
wandb.login()
wandb.init(project=PROJECT_LOG)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.mu_prior_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.mu_prior2_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, mu_prior, next_mu_prior):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.mu_prior_buf[self.ptr] = mu_prior
        self.mu_prior2_buf[self.ptr] = next_mu_prior
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     mu_prior=self.mu_prior_buf[idxs],
                     mu_prior2=self.mu_prior2_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(agents[0].device) for k,v in batch.items()}


class SAC():
    def __init__ (self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, beta=0.3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, use_entropy_loss=True, use_kl_loss=False, use_kl_and_entropy=False, epsilon=1e-5, use_auto_beta=False, use_auto_alpha=False, target_KL_div=0, target_entropy=0.3):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.update_every = update_every
        self.update_after = update_after
        self.steps_per_epoch = steps_per_epoch
        self.use_kl_loss = use_kl_loss
        self.use_kl_and_entropy = use_kl_and_entropy
        self.use_entropy_loss = use_entropy_loss
        self.use_auto_alpha = use_auto_alpha
        self.target_KL_div = target_KL_div
        self.use_auto_beta = use_auto_beta
        self.target_entropy = target_entropy
        self.a_lr = 3e-4
        self.counter = 0

        # CORE-RL Params
        self.factorC = FACTOR_C
        self.lambda_max = LAMBDA_MAX

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up automatic KL div temperature tuning for alpha 
        if self.use_auto_alpha:
            #self.log_alpha = torch.tensor([[-0.7]], requires_grad=True, device=self.device)
            #self.alpha_optim = Adam([self.log_alpha], lr=self.a_lr)
            self.alpha = torch.tensor([[10.0]], requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.alpha], lr=self.a_lr)
        if self.use_auto_beta:
            self.log_beta = torch.tensor([[-0.01]], requires_grad=True, device=self.device)
            self.beta = self.log_beta.exp()
            self.beta_optimizer = Adam([self.log_beta], lr=self.a_lr)


    def compute_loss_q(self, data):
    # Set up function for computing SAC Q-losses    
        o, a, r, o2, d, mu_prior2 = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['mu_prior2']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, mu_policy2, sigma_policy2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            KL_loss = compute_kld_univariate_gaussians(mu_prior2, torch.tensor(sigma_prior).to(self.device), mu_policy2, sigma_policy2).sum(axis=-1)
            
            if self.use_kl_loss:
                # KL minimisation regularisation
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * KL_loss)
            # elif self.use_kl_and_entropy:
            #     backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * KL_loss - self.beta * logp_a2)
            elif self.use_entropy_loss:
                # Maximum entropy backup
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.beta * logp_a2)
                
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, q_pi_targ

    
    def compute_loss_pi(self, data):
    # Set up function for computing SAC pi loss
        o, mu_prior = data['obs'], data['mu_prior']
        pi, logp_pi, mu_policy, sigma_policy = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        KL_loss = compute_kld_univariate_gaussians(mu_prior, torch.tensor(sigma_prior).to(self.device), mu_policy, sigma_policy).sum(axis=-1)

        if self.use_kl_loss:
            # Entropy-regularized policy loss
            loss_pi = (self.alpha * KL_loss - q_pi).mean()
        # elif self.use_kl_and_entropy:
        #     loss_pi = (self.alpha * KL_loss + self.beta * logp_pi - q_pi).mean()
        elif self.use_entropy_loss:
            loss_pi = (self.beta * logp_pi - q_pi).mean()
            
        return loss_pi, logp_pi, KL_loss


    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_pi_targ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, KL_div = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

                
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Update counter
        self.counter += 1


        if self.use_auto_alpha:
            # Update temperature
            #alpha_loss = self.alpha * min(0.0, (self.target_KL_div - KL_div).detach().mean())
            alpha_loss = self.alpha * (self.target_KL_div - KL_div).detach().mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

 
        if self.use_auto_beta:
            # Update temperature
            self.beta_optimizer.zero_grad()
            beta_loss = (-self.log_beta * (self.target_entropy + logp_pi).detach()).mean()
            #beta_loss = (self.beta * (self.target_entropy - logp_pi).detach()).mean()
            beta_loss.backward()
            self.beta_optimizer.step()
            self.beta = self.log_beta.exp()

        
        # Record things
        return {'loss_q': loss_q.item(),
                'loss_pi': loss_pi.item(),
                'entropy': logp_pi.mean().item(),
                'KL_div': KL_div.mean().item(),
                'q_pi_targ_max': q_pi_targ.max(),
                'q_pi_targ_min': q_pi_targ.min(),
                'q_pi_targ_mean': q_pi_targ.mean(),
                'target_KL_div': self.target_KL_div}


    def get_action(self, o, deterministic=False):
        act, mu, std =  self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
        return act, mu, std


#--------------------------------------------------------- Functions -------------------------------------------------------------------------------#

def test_agent(use_single_agent=True):
      
    global total_steps
    global test_steps
    ep_ret = 0.0
    ep_len = 0.0
    
    for j in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        while not (d or (ep_len == agents[0].max_ep_len)):
            
            if METHOD == "residual":
                act_policy, mu, sigma = agents[0].get_action(o, True)
                act_prior = prior.compute_action()
                action = np.clip(act_prior + act_policy, -1, 1)
            else:
                # Take deterministic actions at test time
                if use_single_agent:
                    #agent = random.choice(agents)
                    agent = agents[0]
                    action, mu, sigma = agent.get_action(o, True)
                else:
                    ensemble_actions = ray.get([get_distr.remote(o,p.ac) for p in agents])
                    #mu, sigma = fuse_ensembles_deterministic(ensemble_actions)
                    mu, sigma = fuse_ensembles_stochastic(ensemble_actions)
                    dist = Normal(torch.tensor(mu.detach()), torch.tensor(sigma.detach()))
                    action = torch.tanh(dist.sample()).numpy()
                    wandb.log({'ensemble_std_lin_vel': sigma[0],
                            'ensemble_std_ang_vel': sigma[1],
                            'ensemble_mu_lin_vel': mu[0],
                            'ensemble_mu_ang_vel': mu[1]}, total_steps)

            wandb.log({'test_steps': test_steps}, total_steps)
            o, r, d, _ = agents[0].test_env.step(action)
            ep_ret += r
            ep_len += 1
            total_steps += 1
            test_steps += 1
    avg_ret = ep_ret/agents[0].num_test_episodes
    avg_len = ep_len/agents[0].num_test_episodes
    
    return {'rewards_eval': avg_ret,
            'len_eval': avg_len}

def evaluate_prior_agent():

    ep_ret = 0.0
    ep_len = 0.0

    for  i in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        while not (d or (ep_len == agents[0].max_ep_len)):
            act = prior.compute_action()
            o, r, d, p = env.step(act)
            ep_ret += r
            ep_len += 1
    avg_ret = ep_ret/agents[0].num_test_episodes
    avg_len = ep_len/agents[0].num_test_episodes

    return {'rewards_eval': avg_ret,
            'len_eval': avg_len}
                    
def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma, zeta):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    zeta2 = 1.0-zeta
    mu = (np.power(policy_sigma, 2) * zeta * prior_mu + np.power(prior_sigma,2) * zeta2 * policy_mu)/(np.power(policy_sigma,2) * zeta + np.power(prior_sigma,2) * zeta2)
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma,2) * zeta + np.power(prior_sigma,2) * zeta2))
    return mu, sigma

def inverse_sigmoid_gating_function(k, C, x):
    val = 1 / (1 + math.exp(k*(x - C))) 
    return val

def compute_kld_univariate_gaussians(mu_prior, sigma_prior, mu_policy, sigma_policy):
    # Computes the analytical KL divergence between two univariate gaussians
    kl = torch.log(sigma_policy/sigma_prior) + ((sigma_prior**2 + (mu_prior - mu_policy)**2)/(2*sigma_policy**2)) - 1/2
    return kl

@ray.remote(num_gpus=1)
def get_distr(state, agent):
    state = torch.FloatTensor(state).unsqueeze(0).cuda()
    act, mu, std = agent.act(state, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def fuse_ensembles_deterministic(ensemble_actions):
    actions = torch.tensor([ensemble_actions[i][0] for i in range (NUM_AGENTS)])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

def fuse_ensembles_stochastic(ensemble_actions):
    mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    var = (np.sum(np.array([(ensemble_actions[i][1]**2 + ensemble_actions[i][0]**2)-mu**2 for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    sigma = np.sqrt(var)
    return torch.from_numpy(mu), torch.from_numpy(sigma)

def write_logs(logs, t):
    wandb.log(logs,t)

def save_ensemble():
    for idx, agnt in enumerate(agents):
        torch.save(agnt.ac.pi, save_dir + wandb.run.name + "_" + str(idx) + ".pth")

#--------------------------------------------------------- Run -------------------------------------------------------------------------------#


def run(agents, env):

    # Prepare for interaction with environment
    global total_steps
    o, ep_ret, ep_len = env.reset(), 0, 0
    agent = random.choice(agents)
    r = 0
    o_old = o
    

    # Main loop: collect experience in env and update/log each epoch
    for t in range(NUM_STEPS):

        mu_prior = prior.compute_action()
        
        if t > agent.start_steps:
            policy_action, mu_policy, std_policy = agent.get_action(o)
            write_logs({'mu_policy_1' : mu_policy[0],
                        'mu_policy_2' : mu_policy[1],
                        'std_policy_1': std_policy[0],
                        'std_policy_2': std_policy[1],
                        'mu_prior_1': mu_prior[0],
                        'mu_prior_2': mu_prior[1]}, total_steps)            
            
            if METHOD == "policy":
                a = policy_action

            if METHOD == "CORE-RL":
                with torch.no_grad():
                    act_b, _, _, _ = agent.ac.pi(torch.as_tensor(o_old, dtype=torch.float32).to(agent.device), True, False)
                    base_q  = agent.ac.q1(torch.as_tensor(o_old, dtype=torch.float32).to(agent.device), act_b).cpu().numpy()
                    
                    act_t, _, _, _ = agent.ac.pi(torch.as_tensor(o, dtype=torch.float32).to(agent.device), True, False)
                    target_q = agent.ac.q1(torch.as_tensor(o, dtype=torch.float32).to(agent.device), act_t).cpu().numpy()
                # Compute lambda from measured td-error
                td_error  = (r + agent.gamma * target_q) - base_q
                lambda_mix = agent.lambda_max*(1 - np.exp(-agent.factorC * np.abs(td_error)))
                # Compute the combined action
                a = policy_action/(1+lambda_mix) + (lambda_mix/(1+lambda_mix))*mu_prior
                #a = np.clip(a, -1, 1)
                write_logs({'lambda_mix':lambda_mix}, total_steps)
                write_logs({'td_error':td_error}, total_steps)
                

            if METHOD == "MCF":

                if USE_ENSEMBLE:
                    ensemble_actions = ray.get([get_distr.remote(o,p.ac) for p in agents])
                    #mu_ensemble, sigma_ensemble = fuse_ensembles_deterministic(ensemble_actions)
                    
                    mu_ensemble, sigma_ensemble = fuse_ensembles_stochastic(ensemble_actions)
                    mu_mcf, std_mcf = fuse_controllers(mu_prior, sigma_prior, mu_ensemble.cpu().numpy(), sigma_ensemble.cpu().numpy(), 0.5)
                    
                    write_logs({'std_ensemble_lin_vel':sigma_ensemble[0]}, total_steps)
                    write_logs({'std_ensemble_ang_vel':sigma_ensemble[1]}, total_steps)
                    
                else:
                    zeta = inverse_sigmoid_gating_function(0.00005, 300000, t)
                    mu_mcf, std_mcf = fuse_controllers(mu_prior, sigma_prior, mu_policy.cpu().numpy(), std_policy.cpu().numpy(), zeta)
                    write_logs({'zeta': zeta}, total_steps)

                dist_hybrid = Normal(torch.tensor(mu_mcf).double().detach(), torch.tensor(std_mcf).double().detach())
                a = dist_hybrid.sample()
                a = torch.tanh(a).numpy()
                write_logs({'std_mcf_lin_vel':std_mcf[0]}, total_steps)

            if METHOD == "residual":
                a = np.clip(mu_prior + policy_action, -1, 1)
                
        else:
            a = env.action_space.sample()
            if METHOD == "residual":
                policy_action = a
                a = np.clip(mu_prior + policy_action, -1, 1)

        write_logs({'executed_action':a[0]}, total_steps)
        write_logs({'executed_action2':a[1]}, total_steps)
        
        o2, r, d, _ = env.step(a)
        mu_prior2 = prior.compute_action()
        ep_ret += r
        ep_len += 1
        total_steps +=1
    
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==agent.max_ep_len else d

        # Store experience to replay buffer
        if METHOD == "residual":
            replay_buffer.store(o, policy_action, r, o2, d, mu_prior, mu_prior2)
        else:
            replay_buffer.store(o, a, r, o2, d, mu_prior, mu_prior2)

        o_old = o
        o = o2

        # Update handling
        if t >= agent.update_after:
            for ag in agents:
                batch = replay_buffer.sample_batch(agent.batch_size)
                metrics = ag.update(batch)
            write_logs(metrics, total_steps)
            if agents[0].use_auto_alpha: write_logs({'alpha': agents[0].alpha.cpu().item()}, total_steps)
            if agents[0].use_auto_beta: write_logs({'beta': agents[0].beta.cpu().item()}, total_steps)

        # End of epoch handling
        if (t+1) % agent.steps_per_epoch == 0:
            # Test the performance of the deterministic version of the agent.
            metrics = test_agent(use_single_agent=EVAL_SINGLE_AGENT)
            write_logs(metrics, total_steps)
            save_ensemble()

        # End of trajectory handling
        if d or (ep_len == agent.max_ep_len):
            write_logs({'ep_rewards': ep_ret}, total_steps)
            write_logs({'ep_length': ep_len}, total_steps)

            if ENV == "navigation": env.env_type = np.random.choice([1,2,3,4])
            
            o, ep_ret, ep_len, r = env.reset(), 0, 0, 0
            agent = random.choice(agents)
            
            


TASK = arg("--task", "navigation")
ENV = arg("--env", "PandaReacher" if TASK == "manipulation" else "PointGoalNavigation")
METHOD = arg("--method", "MCF") # Options: policy, MCF, residual
REWARD = arg("--reward", "sparse") 
USE_KL = arg("--use_kl", int(False))
USE_ENTROPY = arg("--use_entropy", int(True))
USE_KL_AND_ENTROPY = arg("--use_kl_and_entropy", int(False))
ALPHA = arg("--alpha", 5 if TASK == "manipulation" else 0.5) #0.01 #0.05 # KL temperature term
BETA = arg("--beta", 0.01 if TASK == "manipulation" else 0.1) #0.01 #0.05 # Entropy temperature term
EPSILON = arg("--epsilon", 2e-4)
SEED = arg("--seed", 0)
NUM_AGENTS = arg("--num_agents", 5)
NUM_STEPS = arg("--num_steps", int(1e6))
USE_ENSEMBLE = arg("--use_ensemble", int(False))
USE_AUTO_ALPHA = arg("--use_auto_alpha", int(False))
USE_AUTO_BETA = arg("--use_auto_beta", int(False))
TARGET_KL_DIV = arg("--target_kl_div", 10e-3)
TARGET_ENTROPY = arg("--target_entropy", -8.0 if TASK == "manipulation" else -7)
SIGMA_PRIOR = arg("--sigma_prior", 0.4 if TASK == "manipulation" else 0.4)
EVAL_SINGLE_AGENT = arg("--eval_single_agent", 1)
PRIOR_METHOD = arg("--prior_method", "APF")
LAMBDA_MAX = arg("--lambda_max", 15.0)
FACTOR_C = arg("--factorC", 0.3)



wandb.run.name = TASK + "_" + "SEED:" + str(SEED) + "_" + METHOD +  "_" "USE_ENSEMBLE:" + str(USE_ENSEMBLE) + "_" + time.asctime().replace(' ', '_')

save_dir = "saved_models/" + wandb.run.name + "/"
os.mkdir(save_dir)

torch.set_num_threads(torch.get_num_threads())
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

wandb.config.update(HYPERS)

if TASK == "manipulation":
    # PyRep
    from manipulation.PandaReacher import ReacherEnv
    from manipulation.prior_controller import RRMC_controller
    max_ep_steps = 1000
    # Swift
    #from manipulation.SwiftPandaReacher import ReacherEnv
    #from manipulation.prior_controller_swift import RRMC_controller
    #max_ep_steps = 225
    
    env = ReacherEnv(headless=True, dense = True if REWARD == "dense" else False, max_ep_steps=max_ep_steps)
    prior = RRMC_controller(env)
    sigma_prior = SIGMA_PRIOR
elif TASK == "navigation":
    from navigation.PointGoalNavigationEnv import PointGoalNavigation
    from navigation.prior_controller import PotentialFieldsController
    from navigation.prior_controller import P_controller
    max_ep_steps = 500
    HYPERS = dict(
        timeout      = max_ep_steps,
        env_type     = 2,
        reward_type  = REWARD)
    for k,v in HYPERS.items(): exec("{} = {!r}".format(k,v))
    env = PointGoalNavigation(**HYPERS)
    if PRIOR_METHOD == "APF":
        prior = PotentialFieldsController(env)
    if PRIOR_METHOD == "P_controller":
        prior = P_controller(env)
    sigma_prior = SIGMA_PRIOR



env.seed(SEED)
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))
total_steps = 0
test_steps = 0


# Initialise an ensemble of agents
agents = [SAC(lambda:env,
            actor_critic=core.MLPActorCritic, 
            gamma=0.99, 
            polyak=0.995, 
            lr=1e-3, 
            batch_size=100, 
            start_steps=100, 
            num_test_episodes=10, 
            max_ep_len=max_ep_steps,
            alpha=ALPHA,
            beta=BETA,
            epsilon=EPSILON,
            use_entropy_loss=USE_ENTROPY,
            use_kl_loss=USE_KL,
            use_kl_and_entropy=USE_KL_AND_ENTROPY,
            use_auto_alpha=USE_AUTO_ALPHA,
            use_auto_beta=USE_AUTO_BETA,
            target_entropy=TARGET_ENTROPY,
            target_KL_div=TARGET_KL_DIV) for _ in range(NUM_AGENTS)]

wandb.watch(agents[0].ac)
run(agents, env)

