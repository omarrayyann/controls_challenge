import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import io
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import clear_output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseController:
  def update(self, target_lataccel, current_lataccel, state, last_action):
    last_road = -99
    def update(self, target_lataccel, current_lataccel, state):
      error = target_lataccel - current_lataccel
      ans = error * 0.3
      return ans
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, last_action):
    return target_lataccel


class SimpleController(BaseController):
  last_road = -99
  def update(self, target_lataccel, current_lataccel, state, last_action):
    error = target_lataccel - current_lataccel
    ans = error * 0.3
    return ans
  

############################
    
class Actor(nn.Module):
    def __init__(self, max_action=2):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(7, 300)
        self.l2 = nn.Linear(300,200)
        self.l3 = nn.Linear(200,200)
        self.l4 = nn.Linear(200, 1)

    def forward(self, s):
        last_action = s[:,-1].view(-1,1)
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        s = F.relu(self.l3(s))
        a = self.max_action * torch.tanh(self.l4(s))  # [-max,max]
        a = a*0.5 + last_action*0.5
        return a

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(8, 300)
        self.l2 = nn.Linear(300,200)
        self.l3 = nn.Linear(200,200)
        self.l4 = nn.Linear(200, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q

class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.hidden_width = 256
        self.batch_size = 128
        self.GAMMA = 0.99
        self.TAU = 0.001
        self.lr = 0.001

        self.actor = Actor().to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor).to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic).to(DEVICE)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.MseLoss = nn.MSELoss().to(DEVICE)

    def save_checkpoint(self, i):
        actor_path = "checkpoints/actor_checkpoint_" + str(i) + ".pth"
        critic_path =  "checkpoints/critic_checkpoint_" + str(i) + ".pth"
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved checkpoints to {actor_path} and {critic_path}")

    def load_checkpoint(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor_target = copy.deepcopy(self.actor).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic).to(DEVICE)
        print(f"Loaded checkpoints from {actor_path} and {critic_path}")

    def choose_action(self, s):
        s = torch.unsqueeze(s.clone().detach().float(), 0).to(DEVICE)
        a = self.actor(s).data.cpu().numpy().flatten()
        return a

    def decrease_learning_rate(self):
        # Reduce learning rate by 10%
        self.lr  = self.lr / 10
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr

        print(f"Updated learning rate to {self.lr}")

    def learn(self, relay_buffer):

        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  
        batch_s = batch_s.to(DEVICE)
        batch_r = batch_r.to(DEVICE)
        batch_s_ = batch_s_.to(DEVICE)
        batch_dw = batch_dw.to(DEVICE)
        batch_a = batch_a.to(DEVICE)

        with torch.no_grad(): 
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_)).to(DEVICE)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q,target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)




class MyController(BaseController):
  last_road = -99

  state_dim = 7
  action_dim = 1
  max_action = 2
  agent = DDPG(state_dim, action_dim)

  agent.load_checkpoint("checkpoints/actor_checkpoint_40.98172332648468.pth","checkpoints/critic_checkpoint_40.98172332648468.pth")

  def update(self, target_lataccel, current_lataccel, state,last_action):
    roll_lataccel, v_ego, a_ego = state
    state = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, target_lataccel-current_lataccel, last_action])
    action = self.agent.choose_action(state)[0]

    error = target_lataccel-current_lataccel
    factor = 0.8
    
    return action*(1.0-factor) + factor*last_action
  
###################



CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'mine': MyController,
}

