# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
 
import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import TQC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from framestack import FrameStack, mkdir
import torchvision.transforms.functional as TF




def evaluate_policy(policy, writer, total_timesteps, size, env, episode=5):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """

    path = mkdir("","eval/" + str(total_timesteps) + "/")
    print(path)
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    goal= 0
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        done = False
        for step in range(50):
            action = policy.select_action(np.array(obs))
            
            obs, reward, done, info = env.step(action)
            #img = cv2.resize(img.astype(np.float), (300,300))
            #cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            #frame = cv2.imwrite("{}/wi{}.png".format(path, step), np.array(img))
            if done:
                avg_reward += reward 
                if step < 49:
                    goal +=1
                break
            #cv2.waitKey(10)
            avg_reward += reward 

    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {}  goal reached {} of  {} ".format(avg_reward, goal, episode))
    print ("---------------------------------------")
    return avg_reward




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def time_format(sec):
    """
    
    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train_agent(config):
    """

    Args:
    """
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    file_name = str(config["locexp"]) + "/pytorch_models/{}".format(str(config["env_name"]))    
    pathname = dt_string 
    tensorboard_name = str(config["locexp"]) + pathname + '/runs/' + pathname
    print(tensorboard_name)
    writer = SummaryWriter(tensorboard_name)
    size = config["size"]
    print("create env ..")
    env = gym.make(config["env_name"], renderer='egl')
    #env = gym.make(config["env_name"])
    print("... done")
    env = FrameStack(env, config)
    print("...   framdone")
    obs = env.reset()
    print("...   reset done")
    print("state ", obs.shape)
    state_dim = 512
    print("State dim, " , state_dim)
    action_dim = 5 
    print("action_dim ", action_dim)
    max_action = 1
    config["target_entropy"] =-np.prod(action_dim)
    #obs_shape = (size, size, config["history_length"])
    obs_shape = (config["history_length"], size, size)
    action_shape = (action_dim,)
    print("obs", obs_shape)
    print("act", action_shape)
    policy = TQC(state_dim, action_dim, max_action, config)    
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(config["buffer_size"]), config["image_pad"], config["device"])
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = 0
    evaluations = []
    tb_update_counter = 0
    # evaluations.append(evaluate_policy(policy, writer, total_timesteps, size, env))
    # save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
    # policy.save(save_model)
    done_counter =  deque(maxlen=100)
    while total_timesteps <  config["max_timesteps"]:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            #env.seed(random.randint(0, 100))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > config["tensorboard_freq"]:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                if episode_timesteps < 50:
                    done_counter.append(1)
                else:
                    done_counter.append(0)
                goals = sum(done_counter)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                writer.add_scalar('Goal_freq', goals, total_timesteps)
                
                print(text)
                write_into_file(pathname, text)
            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= config["eval_freq"]:
                timesteps_since_eval %= config["eval_freq"]
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, size,  env))
                save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
                print("Save model to {}".format(save_model))
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            obs = env.reset()
            first_image = True
            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < config["start_timesteps"]:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(obs, replay_buffer)
        
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        # print("shape obs", new_obs.shape)
        #frame = cv2.imshow("wi", np.array(new_obs))
        #cv2.waitKey(10)
        done = float(done)
        
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        done_bool = 0 if episode_timesteps + 1 == config["max_episode_steps"] else float(done)
        if episode_timesteps + 1 == config["max_episode_steps"]:
            done = True
        # We increase the total reward
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        if episode_timesteps == 0:
            replay_buffer.add(obs, action, reward, new_obs, done, done_bool, first=True)
        elif episode_timesteps == 1:
            replay_buffer.add(obs, action, reward, new_obs, done, done_bool, second=True)
        else:
            replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > config["start_timesteps"]:
            for i in range(1):
                policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
