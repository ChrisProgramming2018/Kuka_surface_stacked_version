import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid


class Actor(Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.device = config["device"]
        self.log_std_min_max = (-20, 2)
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)
    
    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*self.log_std_min_max)
        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std, self.device)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob
    
    def select_action(self, obs):
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action



class Critic(Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.nets = []
        self.n_quantiles = config["n_quantiles"]
        self.n_nets =  config["n_nets"]
        for i in range(self.n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], self.n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles

class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, device):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device), torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)
    
    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result
    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh

class Mlp(Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output

class Encoder(nn.Module):
    def __init__(self, config, D_out=512, conv_channels=[32, 64, 64], kernel_sizes=[4, 4, 3], strides=[2, 1, 1]):
        super(Encoder, self).__init__()
        # Defining the first Critic neural network
        #channels = config["history_length"]
        channels = 24
        self.conv_1 =  torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0])
        self.conv_2 =  torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1])
        self.conv_3 =  torch.nn.Conv2d(conv_channels[1], conv_channels[2], kernel_sizes[2], strides[2])
        self.flatten = torch.nn.Flatten()
        self.Linear  = torch.nn.Linear(1024, 512)

    def create_vector(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])  
        x = self.conv_1(obs)
        x = self.conv_2(x)
        x = self.flatten(x)
        # print(x.shape)
        obs = self.Linear(x)

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs


def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

