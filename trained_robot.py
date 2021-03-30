import os
import sys
import gym
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from agent import TQC
import cv2
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from replay_buffer import ReplayBuffer
from helper import FrameStack


def evaluate_policy(policy, args, env, episode=25):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    size = args.size
    different_seeds = args.seed
    seeds = [x for x in range(episode)]
    obs_shape = (args.history_length, size, size)
    action_shape = (args.action_dim,)
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(args.buffer_size), args.image_pad, args.device)
    path = "eval"
    goals  = 0
    avg_reward = 0.
    total_steps = 0
    for s in seeds:
        if different_seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            env.seed(s)
        obs = env.reset()
        done = False
        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            new_obs, reward, done, _ = env.step(action)
            frame = cv2.imwrite("{}/{}wi{}.png".format(path, s, step), np.array(new_obs)[:,:,[2,1,0]])
            done_bool = 0 if step + 1 == args.max_episode_steps else float(done)
            replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
            if done:

                if step < 49:
                    total_steps += step 
                    goals +=1
                break
            obs = new_obs
            avg_reward += reward * args.reward_scalling


    if args.save_buffer:
        replay_buffer.save_memory("/export/leiningc/")
    avg_reward /= len(seeds)
    print("reached goal {} of {}".format(goals, episode))
    if goals != 0:
        print("Average step if reached {} ".format(float(total_steps) / goals))
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {} of {} Episode".format(avg_reward, len(seeds)))
    print ("---------------------------------------")
    return avg_reward




def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    env= gym.make(args.env_name, renderer='egl')
    env = FrameStack(env, args)
    print(env.action_space)
    state = env.reset()
    state_dim = 200
    action_dim = env.action_shape[0]
    args.action_dim = action_dim
    max_action = float(1)
    min_action = float(-1)
    args.target_entropy=-np.prod(action_dim)
    
    policy = TQC(state_dim, action_dim, max_action, args) 
    directory = "pretrained/"
    if args.agent is None:
        filename= "kuka_block_grasping-v0-47025reward_-0.87-agentTCQ"
        filename = "kuka_block_grasping-v0-34433reward_-0.86-agentTCQ"
        filename = "kuka_block_grasping-v0-79900reward_-0.86-agentTCQ"
        filename= "kuka_block_grasping-v0-43240reward_-0.99-agentTCQ"
        filename= "kuka_block_grasping-v0-43240reward_-0.99-agentTCQ"
        filename= "kuka_block_grasping-v0-76721reward_-0.88-agentTCQ"  # 91 % 
        filename = "kuka_block_grasping-v0-97133reward_-1.05-agentTCQ" # 93 %
    else:
        filename = args.agent

    filename = directory + filename
    print("Load " , filename)
    policy.load(filename)
    policy.actor.training = False
    if args.eval:
        evaluate_policy(policy, args,  env, args.epi)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=25, type=int)
    parser.add_argument('--max_episode_steps', default=50, type=int)    
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_decoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)     #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--eval', type=bool, default= True)
    parser.add_argument('--buffer_size', default=3.5e5, type=int)
    arg = parser.parse_args()
    main(arg)
