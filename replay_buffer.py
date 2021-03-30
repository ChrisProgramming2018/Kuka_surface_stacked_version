import numpy as np
import kornia
import  sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from taskonomy_network import TaskonomyNetwork


np.set_printoptions(threshold=np.inf)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.bool)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.bool)
        self.model = TaskonomyNetwork().to(device)
        self.model.load_model("trained_models/model-21170.39453125")
        self.idx = 0
        self.full = False
        self.first_idx = {}
        self.sec_idx = {}

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, first=False, second=False):
        # print("add shape", obs[0].shape)

        if first:
            self.first_idx.update({self.idx:True})
        if second:
            self.sec_idx.update({self.idx:True})
        np.copyto(self.obses[self.idx], obs[0].transpose(2,0,1))     # save only the current image
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[0].transpose(2,0,1))
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = (self.idx + 1) % self.capacity - 2
        self.full = self.full or self.idx == self.capacity - 2  # keeps the last two spots as for stacking


    def sample(self, batch_size):
        """      """
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        all_idxs = []
        for i in idxs:
            # case i is first 
            if i in self.first_idx:
                all_idxs.append(-2)
                all_idxs.append(-1)
                all_idxs.append(i)
            elif i in self.sec_idx:
                # case i is second
                all_idxs.append(-2)
                all_idxs.append(i-1)
                all_idxs.append(i)
            else:
                all_idxs.append(i-2)
                all_idxs.append(i-1)
                all_idxs.append(i)
        # all_idxs = list(set(all_idxs))
        #print("all ", all_idxs)
        #print("s ", idxs)
        obses = self.obses[all_idxs]
        next_obses = self.next_obses[all_idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        
        obses = self.aug_trans(obses).div_(255)
        next_obses = self.aug_trans(next_obses).div_(255)
        obses_aug = self.aug_trans(obses).div_(255)
        next_obses_aug = self.aug_trans(next_obses).div_(255)
        obses = self.model.encoder(obses)
        next_obses = self.model.encoder(next_obses)
        obses_aug = self.model.encoder(obses_aug)
        next_obses_aug = self.model.encoder(next_obses_aug)
        obses = self.stack(obses)
        next_obses = self.stack(next_obses)
        obses_aug = self.stack(obses_aug)
        next_obses_aug = self.stack(next_obses_aug)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug

    def stack(self, obses):
        """ stack every 3 to one   """
        states = []
        temp = []
        for idx, obs in enumerate(obses):
            temp.append(obs)
            if (idx + 1) % 3 == 0:
                s = torch.cat(temp)
                states.append(s.unsqueeze(0))
                temp = []
        obs = torch.cat(states)
        return obs


    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)
        
        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)
        
        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)
        
        with open(filename + '/not_dones.npy', 'rb') as f:
            self.not_dones = np.load(f)
        
        with open(filename + '/not_dones_no_max.npy', 'rb') as f:
            self.not_dones_no_max = np.load(f)
