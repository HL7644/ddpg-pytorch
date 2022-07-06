import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
import collections
import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install tensorboardX
from tensorboardX import SummaryWriter

from policy_module import *
from value_module import *

Ep_Step=collections.namedtuple('EpStep', field_names=['obs', 'action','reward','obs_f','termin_signal'])

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, obs):
    obs=torch.FloatTensor(obs).to(device)
    return obs

class Replay_Buffer(torch.utils.data.Dataset):
  def __init__(self):
    super(Replay_Buffer, self).__init__()
    self.ep_steps=[]
    self._max_length=100000
    self._length=0
  
  def sample_batch(self, batch_size):
    batch=[]
    batch_idx=np.random.choice(self._length, batch_size)
    for idx in batch_idx:
      batch.append(self.ep_steps[idx])
    return batch
  
  def add_item(self, ep_step):
    if self._length>=self._max_length:
      #remove earliest element
      self.ep_steps.pop(0)
      self._length=self._length-1
    #add element
    self.ep_steps.append(ep_step)
    self._length=self._length+1
    return
  
  def __getitem__(self, idx):
    return self.ep_steps[idx]
  
  def __len__(self):
    return len(self.ep_steps)

class DDPG_Agent():
  def __init__(self, env, test_env, gamma, lambd):
    self.env=env
    self.test_env=test_env
    self.gamma=gamma
    self.lambd=lambd
    #both are box class
    i_size=self.env.observation_space.shape[0]
    a_dim=self.env.action_space.shape[0]
    self.a_low=torch.FloatTensor(self.env.action_space.low).to(device)
    self.a_high=torch.FloatTensor(self.env.action_space.high).to(device)

    self.pm=Cont_Deterministic_Policy_Module(i_size, [400,300], a_dim)
    #use same parameters as pm
    self.target_pm=Cont_Deterministic_Policy_Module(i_size, [400,300], a_dim)
    pm_params=self.pm.vectorize_parameters()
    self.target_pm.inherit_parameters(pm_params)

    self.vm=DDPG_Value_Module(i_size, a_dim, [400,300])
    #use same parameters as vm
    self.target_vm=DDPG_Value_Module(i_size, a_dim, [400,300])
    vm_params=self.vm.vectorize_parameters()
    self.target_vm.inherit_parameters(vm_params)

  def add_noise(self, action, std):
    dim_a=action.size(0)
    noise=torch.normal(torch.zeros(dim_a), torch.full([dim_a], std)).to(device)
    action=action+noise
    return action

  def get_action(self, obs, std, a_mode, train=True):
    if a_mode=='random':
      action=torch.FloatTensor(self.env.action_space.sample()).to(device)
    else:
      action=self.pm(obs)
      if train:
        action=self.add_noise(action, std)
    action=torch.clamp(action, self.a_low, self.a_high)
    return action

writer=SummaryWriter(logdir='ddpg')

class DDPG():
  def __init__(self, agent):
    self.agent=agent
    #replay buffer
    self.replay_buffer=Replay_Buffer()
  
  def get_value_loss(self, batch_data): #critic
    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for ep_step in batch_data:
      obs=ep_step.obs
      action=torch.FloatTensor(ep_step.action).to(device)
      reward=ep_step.reward
      obs_f=ep_step.obs_f
      termin_signal=ep_step.termin_signal

      Q=self.agent.vm(obs, action)
      #create target using target networks, targets doesn't need gradient
      target_action_f=self.agent.target_pm(obs_f)
      target=(reward+self.agent.gamma*(1-termin_signal)*self.agent.target_vm(obs_f, target_action_f)).detach()
      value_loss=value_loss+(target-Q)**2
    value_loss=value_loss/batch_size
    return value_loss
  
  def get_policy_loss(self, batch_data): #actor: using deterministic policy gradient -> perform SGA
    batch_size=len(batch_data)
    policy_loss=torch.FloatTensor([0]).to(device)
    for ep_step in batch_data:
      obs=ep_step.obs
      reward=ep_step.reward
      termin_signal=ep_step.termin_signal
      #data: sampled from another policy
      action=self.agent.pm(obs)
      #action for loss: computed from current policy (the one we're updating) => off-policy learning
      Q=self.agent.vm(obs, action)
      policy_loss=policy_loss-Q
    policy_loss=policy_loss/batch_size
    return policy_loss
  
  def check_performance(self):
    #w.r.t test env.: run 10 episodes
    len_eps=[]
    acc_rews=[]
    ep_datas=[]
    for _ in range(10):
      obs=self.agent.test_env.reset()
      len_ep=1
      acc_rew=0
      ep_data=[]
      while True:
        action=self.agent.pm(obs)
        action=torch.clamp(action, self.agent.a_low, self.agent.a_high).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        ep_step=Ep_Step(obs, action, reward, obs_f, termin_signal)
        ep_data.append(ep_step)
        acc_rew+=reward
        len_ep+=1
        obs=obs_f
        if termin_signal:
          break
      len_eps.append(len_ep)
      acc_rews.append(acc_rew)
      ep_datas.append(ep_data)
    avg_acc_rew=sum(acc_rews)/10
    avg_len_ep=sum(len_eps)/10
    return avg_acc_rew, avg_len_ep, ep_datas
    
  def train(self, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, 
            act_noise, p_lr, v_lr, polyak): #polyak for delayed target network updates
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim=optim.Adam(self.agent.vm.parameters(), lr=v_lr)

    obs=self.agent.env.reset()
    a_mode='random' #initially random
    update=False
    step=1

    for epoch in range(1, n_epochs+1):
      #"steps_per_epoch" steps per epoch
      while True:
        if step>update_after: #begin update
          update=True
        if step>start_after: #behave w.r.t policy
          a_mode='policy'

        action=self.agent.get_action(obs, act_noise, a_mode).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.env.step(action)
        if termin_signal: #termin signal and location at goal
          if obs_f[0]<0.45:
            termin_signal=0
          else:
            termin_signal=1
          obs_f=self.agent.env.reset()
        else:
          termin_signal=0
        ep_step=Ep_Step(obs, action, reward, obs_f, termin_signal)
        #add to R
        self.replay_buffer.add_item(ep_step)
        if update and step%update_every==1:
          #take "update_every" steps of update -> update target networks
          #create batch data to work with
          batch_data=self.replay_buffer.sample_batch(batch_size)
          for u_step in range(1, update_every+1): #update as many as delayed steps ex) wait 50 steps between updates: take 50 update steps
            ddpg_value_loss=self.get_value_loss(batch_data)
            value_optim.zero_grad()
            ddpg_value_loss.backward()
            value_optim.step()
            #writer.add_scalar('Value Loss Step: {:d}'.format(step), ddpg_value_loss.item(), u_step)

            ddpg_policy_loss=self.get_policy_loss(batch_data)
            policy_optim.zero_grad()
            ddpg_policy_loss.backward()
            policy_optim.step()
            #writer.add_scalar('Policy Loss Step: {:d}'.format(step), ddpg_policy_loss.item(), u_step)
            #print(step, u_step)
            #print(ddpg_value_loss.item(), ddpg_policy_loss.item())
            
            #update target networks
            value_param=self.agent.vm.vectorize_parameters()
            target_value_param=self.agent.target_vm.vectorize_parameters()
            new_target_value_param=polyak*target_value_param+(1-polyak)*value_param
            self.agent.target_vm.inherit_parameters(new_target_value_param)

            policy_param=self.agent.pm.vectorize_parameters()
            target_policy_param=self.agent.target_pm.vectorize_parameters()
            new_target_policy_param=polyak*target_policy_param+(1-polyak)*policy_param
            self.agent.target_pm.inherit_parameters(new_target_policy_param)
        obs=obs_f
        step=step+1
        if step%steps_per_epoch==1:
          break
      #present per-epoch data
      avg_acc_rew, avg_len_ep,_=self.check_performance()
      print("Epoch: {:d}, Avg_Return {:.3f}, Avg Ep Length: {:.2f}".format(epoch, avg_acc_rew, avg_len_ep))
      writer.add_scalar('Avg_Return', avg_acc_rew, epoch)
      writer.add_scalar('Avg Ep Length', avg_len_ep, epoch)
    return

cont_mc=Obs_Wrapper(gym.make('MountainCarContinuous-v0')) #implemented with continuous mountain-car task
test_env=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
agent=DDPG_Agent(cont_mc, test_env, 0.99, 0.97)
ddpg=DDPG(agent)

ddpg.train(batch_size=64, n_epochs=100, steps_per_epoch=4000, start_after=10000, update_after=1000,
           update_every=50, act_noise=0.1, p_lr=1e-3, v_lr=1e-3, polyak=.999)