import random
import numpy as np
from collections import deque,namedtuple

import torch
from torch import nn,optim
import torch.nn.functional as F

from model import Actor,Critic

# Hyperparameters
batch_size = 128
gamma = 0.99
buffer_size = int(1e5)
TAU = 1e-3
lr_actor = lr_critic1 = lr_critic2 = 1e-3
noise_clip = 0.5
start_length = 128
exploration_noise = 0.1
policy_noise = 0.2
policy_delay = 2   

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self,state_size,action_size,max_action,num_agents,seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.max_action = max_action
        self.num_agents = num_agents
        
        self.actor_local = Actor(state_size,action_size,seed,max_action).to(device)
        self.actor_target = Actor(state_size,action_size,seed,max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=lr_actor)
        
        self.critic1_local = Critic(state_size,action_size,seed).to(device)
        self.critic1_target = Critic(state_size,action_size,seed).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(),lr=lr_critic1)
        
        self.critic2_local = Critic(state_size,action_size,seed).to(device)
        self.critic2_target = Critic(state_size,action_size,seed).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(),lr=lr_critic2)
        
        self.memory = REPLAYBUFFER(action_size,batch_size,buffer_size,seed)
        self.t_step = 0
    
    def step(self,states,actions,rewards,next_states,dones):
        for i in range(self.num_agents):
            self.memory.add(states[i,:],actions[i,:],rewards[i],next_states[i,:],dones[i])
        
        if len(self.memory) > start_length:
            experiences = self.memory.sample()
            self.learn(experiences,gamma)
    
    def act(self,states,noise=True):
        states = torch.from_numpy(states).float().to(device)
        
        actions = np.zeros((self.num_agents,self.action_size))
        
        self.actor_local.eval()
        with torch.no_grad():
            for idx,state in enumerate(states):
                actions[idx] = self.actor_local(state).cpu().data.numpy()
                
                # noise added to encourage exploration
                if noise:
                    actions[idx] += np.random.normal(0,exploration_noise,size=self.action_size)
        self.actor_local.train()
        
        return actions.clip(-self.max_action,self.max_action)
    
    def learn(self,experiences,gamma):
        states,actions,rewards,next_states,dones = experiences
        
        next_actions = self.actor_target(next_states) 
        noise = torch.normal(torch.zeros(next_actions.size()),policy_noise).to(device)
        noise = noise.clamp(-noise_clip,noise_clip)
        
        next_actions = (next_actions + noise).clamp(-self.max_action,self.max_action)
        
        # clipped double Q learning
        Q_tar_critic1 = self.critic1_target(next_states,next_actions)
        Q_tar_critic2 = self.critic2_target(next_states,next_actions)
        Q_tar_critic = torch.min(Q_tar_critic1,Q_tar_critic2)
        
        Q_tar = rewards + (gamma*Q_tar_critic*(1-dones)).detach()
        
        # optimize critic 1
        Q_est_critic1 = self.critic1_local(states,actions)
        critic1_loss = F.mse_loss(Q_est_critic1,Q_tar)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # optimize critic 2
        Q_est_critic2 = self.critic2_local(states,actions)
        critic2_loss = F.mse_loss(Q_est_critic2,Q_tar)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # actor update
        
        # update policy every 2 time steps
        self.t_step = (self.t_step + 1) % policy_delay
        if self.t_step == 0:
            
            actions_est = self.actor_local(states)
            
            # Pytorch does gradient descent by default we need gradient ascent
            actor_loss = -self.critic1_local(states,actions_est).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.soft_update(self.actor_local,self.actor_target,TAU)
            self.soft_update(self.critic1_local,self.critic1_target,TAU)
            self.soft_update(self.critic2_local,self.critic2_target,TAU)
    
    def soft_update(self,local,target,tau):
        for lp,tp in zip(local.parameters(),target.parameters()):
            tp.data.copy_(lp.data * tau + tp.data * (1-tau))

            
class REPLAYBUFFER:
    def __init__(self,action_size,batch_size,buffer_size,seed):
        
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
        self.seed  = random.seed(seed)
    
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory,self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        return len(self.memory)