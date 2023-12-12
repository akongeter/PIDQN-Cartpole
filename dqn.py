"""
Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html by Adam Paszke

BSD 3-Clause License

Copyright (c) 2017-2022, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from itertools import count
import math
import random
from collections import namedtuple, deque
import datetime

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'prev_action'))


class ReplayMemory(object):

    def __init__(self, capacity: int, persistence: float = 0):
        self.memory = deque([], maxlen=capacity)
        self.start = []
        self.persistence = persistence
        self.capacity = capacity
        assert 0 <= self.persistence <= 1

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        if self.persistence > 0 and len(self.start) < int(self.capacity * self.persistence):
            self.start.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample([*self.memory, *self.start], batch_size)

    def set_persistence(self, persistence):
        self.persistence = persistence

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, env: gym.Env, tau: float = 0.005, lr: float = 1e-4, batch_size: int = 128, gamma: float = 0.99, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 1000, n_hid: [int] = None, memory_size: int = 10000, train_freq: int = 1, train_start: int = 0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.TAU = tau
        self.LR = lr
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.n_hid = n_hid
        if n_hid is None:
            self.n_hid = [256, 128, 256]
        observations, _ = env.reset()
        self.num_actions = env.action_space.n
        self.num_observations = len(observations)
        self.prev_action = torch.tensor([[0]], device=self.device)
        self.target_net = DQNModel(self.num_actions, self.num_observations, self.n_hid).to(self.device)
        self.policy_net = DQNModel(self.num_actions, self.num_observations, self.n_hid).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.amount_of_memory = memory_size
        self.memory = ReplayMemory(self.amount_of_memory)

        self.steps_done = 0
        self.episode_durations = []
        self.total_timesteps = 0

        self.start_time = datetime.datetime.now()

        self.train_freq = train_freq
        self.train_start = train_start

        self.last_average = 0
        self.best_average = 0
        self.reset_average = 0
        self.best_target_weights = {}
        self.best_policy_weights = {}
        self.second_best_target_weights = {}
        self.second_best_policy_weights = {}

        self.reporting_freq = 100

        self.tried = 0

    def curr_algorithm(self):
        return "DQN"

    def save(self, filename: str = "log.txt"):
        ct = datetime.datetime.now()
        log_filename = filename + "-log.txt"
        with open(log_filename, "w") as f:
            f.write("Algorithm: "+ self.curr_algorithm() + "\n")
            f.write("Start Time: " + str(self.start_time) + "\n")
            f.write("End Time: " + str(ct) + "\n")
            f.write("Parameters: tau; lr; batch_size; gamma; eps_start; eps_end; eps_decay, timesteps, "
                    "episode_durations, optimizer, amount_of_replay_memory \n")
            f.write("tau: " + str(self.TAU) + "\n")
            f.write("lr: " + str(self.LR) + "\n")
            f.write("batch_size: " + str(self.BATCH_SIZE) + "\n")
            f.write("gamma: " + str(self.GAMMA) + "\n")
            f.write("eps_start: " + str(self.EPS_START) + "\n")
            f.write("eps_end: " + str(self.EPS_END) + "\n")
            f.write("eps_decay: " + str(self.EPS_DECAY) + "\n")
            f.write("timesteps: " + str(self.total_timesteps) + "\n")
            f.write("episode_durations: " + str(self.episode_durations) + "\n")
            f.write("optimizer: AdamW \n")
            f.write("amount_of_replay_memory: " + str(self.amount_of_memory) + "\n")
            f.write("persistence: " + str(self.memory.persistence) + "\n")
        model_filename = filename + "-model.txt"
        with open(model_filename, "wb") as f:
            pickle.dump(self, f)

    def status_report(self):
        print("Now training for " + str(len(self.episode_durations)) + " episodes")
        print("Last rewards were  " + str(self.episode_durations[-self.reporting_freq:]))
        average = sum(self.episode_durations[-self.reporting_freq:])/self.reporting_freq
        print(f'Average: {average}')
        print(f'delta: {average - self.last_average}')
        # self.last_average = average
        print(f'Lived through {sum(self.episode_durations)} timesteps and remembering {len(self.memory)} of them')

    def call_model(self, model_name, states):
        if model_name == "target":
            model = self.target_net(states)
        elif model_name == "policy":
            model = self.policy_net(states)
        else:
            raise ValueError("Not a current network!")
        return model

    def call_batch_model(self, model_name, states, prev_actions):
        return self.call_model(model_name, states)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        non_final_actions = torch.cat([batch.prev_action[i] for i in range(len(non_final_mask)) if non_final_mask[i]])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        prev_action_batch = torch.cat(batch.prev_action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.call_batch_model("policy", state_batch, prev_action_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.call_batch_model("target", non_final_next_states, non_final_actions).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def learn(self, total_timesteps: int = 100) -> 'DQN':
        self.total_timesteps = total_timesteps
        for timestep in range(total_timesteps):
            obs = self.env.reset()
            observations, info = obs
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.predict_(observations)[0]
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(observations, action, next_state, reward, self.prev_action)

                observations = next_state
                so_far = sum(self.episode_durations) + t
                if so_far > self.train_start and so_far % self.train_freq == 0:
                    self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                                1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                if done:
                    self.episode_durations.append(t + 1)
                    if len(self.episode_durations) % self.reporting_freq == 0 and len(self.episode_durations) > 0:
                        self.status_report()
                        # forget the weights of the last 50/100 episodes if average is going down too much
                        average = sum(self.episode_durations[-self.reporting_freq:]) / self.reporting_freq
                        if average > self.best_average:
                            print("Saving new best average")
                            self.best_average = average
                            for name, param in self.policy_net.named_parameters():
                                self.best_policy_weights[name] = param.data
                            for name, param in self.target_net.named_parameters():
                                self.best_target_weights[name] = param.data
                            print("Average saved")

                        if average < self.last_average - 10.0 and average < self.best_average - 5.0:
                            print("Resetting Weights")
                            self.tried += 1
                            if average > self.reset_average:
                                for name, param in self.policy_net.named_parameters():
                                    self.second_best_policy_weights[name] = param.data
                                for name, param in self.target_net.named_parameters():
                                    self.second_best_target_weights[name] = param.data
                            for name, param in self.policy_net.named_parameters():
                                param.data = self.best_policy_weights[name]
                            for name, param in self.target_net.named_parameters():
                                param.data = self.best_target_weights[name]
                        elif self.tried == 10:
                            print("Take best reset")
                            for name, param in self.policy_net.named_parameters():
                                param.data = self.second_best_policy_weights[name]
                            for name, param in self.target_net.named_parameters():
                                param.data = self.second_best_target_weights[name]
                        else:
                            self.tried = 0
                            self.reset_average = 0
                            self.last_average = average



                    break
        return self

    def predict_(self, obs):
        observations = obs
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.call_model("policy", observations).max(1)[1].view(1, 1), np.zeros(5)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long), np.zeros(5)

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, state = self.predict_(obs)
        return action.detach().cpu().item(), state

    def set_environment(self, env):
        self.env = env

    @staticmethod
    def load(filename: str, env: gym.Env = None) -> 'DQN':
        model_filename = filename + "-model.txt"
        with open(model_filename, "rb") as f:
            dqn = pickle.load(f)
        if env is not None:
            dqn.set_environment(env)
        return dqn


class DQNModel(nn.Module):
    def __init__(self, num_actions: int, num_observations: int, n_hid):
        super(DQNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_observations, n_hid[0]))
        for i in range(1, len(n_hid)):
            self.layers.append(nn.Linear(n_hid[i-1], n_hid[i]))
        self.layers.append(nn.Linear(n_hid[-1], num_actions))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return F.relu(self.layers[-1](x))

