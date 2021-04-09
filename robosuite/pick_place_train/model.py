import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import glob

import numpy as np
import ipdb
# from torchsummary import summary

# from pushover import notify
# from utils import makegif
from random import randint


class ContinuousActionLinearPolicy(object):
	def __init__(self, theta, state_dim, action_dim):
		assert len(theta) == (state_dim + 1) * action_dim
		self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
		self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)
		self.state_dim = state_dim
		self.action_dim = action_dim
	def act(self, state):
		# a = state.dot(self.W) + self.b
		# ipdb.set_trace()
		a = np.dot(state, self.W) + self.b
		return a
	def update(self, theta):
		self.W = theta[0 : self.state_dim * self.action_dim].reshape(self.state_dim, self.action_dim)
		self.b = theta[self.state_dim * self.action_dim : None].reshape(1, self.action_dim)


class CEM():
	''' cross-entropy method, as optimization of the action policy 
	the policy weights theta are stored in this CEM class instead of in the policy
	'''
	def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
		self.theta_dim = theta_dim
		self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

		
	def sample(self):
		# theta = self.mean + np.random.randn(self.theta_dim) * self.std
		theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
		return theta

	def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
		self.mean = ini_mean_scale*np.ones(self.theta_dim)
		self.std = ini_std_scale*np.ones(self.theta_dim)

	def sample_multi(self, n):
		theta_list=[]
		for i in range(n):
			theta_list.append(self.sample())
		return np.array(theta_list)


	def update(self, selected_samples):
		self.mean = np.mean(selected_samples, axis = 0)
		# print('mean: ', self.mean)
		self.std = np.std(selected_samples, axis = 0) # plus the entropy offset, or else easily get 0 std
		# print('mean std: ', np.mean(self.std))

		return self.mean, self.std



class QNetwork(nn.Module):
	def __init__(self):
		super(QNetwork, self).__init__()

		self.image_channels = 3

		self.action_state_network = nn.Sequential(
										nn.Linear(9,256),
										nn.ReLU(),
										nn.Linear(256,64),
										nn.ReLU(),
										)

		self.image_network = nn.Sequential(
								nn.Conv2d(self.image_channels,8, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(8, 16, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(16, 32, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(32, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU(),
								nn.Conv2d(64, 64, kernel_size=2, stride=2),
								nn.ReLU())

		self.combined_network = nn.Sequential(
								nn.Linear(128,32),
								nn.ReLU(),
								nn.Linear(32,8),
								nn.ReLU(),
								nn.Linear(8,1),
								nn.Sigmoid())

	def forward(self,state,action,image):
		image = image.permute(0,3,1,2)
		x1 = self.action_state_network(torch.cat((state,action),dim=1))
		batch_size,layer_size = x1.shape
		x1 = x1.reshape(batch_size,layer_size,1,1)
		x2 = self.image_network(image)
		x = torch.cat((x1,x2),dim=1)
		x = x.view(x.size(0), -1)
		x = self.combined_network(x)

		return x,x2




class QT_Opt():
	def __init__(self, replay_buffer, qnet, target_qnet1, target_qnet2, 
										hidden_dim=64, q_lr=3e-4, cem_update_itr=4, select_num=6, num_samples=16):
		
		self.state_dim = 66
		self.action_dim = 7
		self.device = torch.device('cpu', 0)
		self.num_samples = num_samples
		self.select_num = select_num
		self.cem_update_itr = cem_update_itr
		self.replay_buffer = replay_buffer
		self.qnet = qnet.to(self.device) # gpu
		self.target_qnet1 = target_qnet1.to(self.device)
		self.target_qnet2 = target_qnet2.to(self.device)
		self.cem = CEM(theta_dim = (self.state_dim + 1) * self.action_dim)  # cross-entropy method for updating
		theta = self.cem.sample()
		self.policy = ContinuousActionLinearPolicy(theta, self.state_dim, self.action_dim)

		self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
		self.step_cnt = 0

	def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=100):
		state, action, reward, next_state, image_current, image_next, done = self.replay_buffer.sample(batch_size)
		self.step_cnt+=1

		
		state_      = torch.FloatTensor(state).to(self.device)
		next_state_ = torch.FloatTensor(next_state).to(self.device)
		action     = torch.FloatTensor(action).to(self.device)
		image_current = torch.FloatTensor(image_current).to(self.device)
		image_next = torch.FloatTensor(image_next).to(self.device)
		reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
		done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

		predict_q, _ = self.qnet(state_, action, image_current) # predicted Q(s,a) value
		_, feature_vector_next = self.qnet(next_state_, action, image_next) # get feature vector for next state
		feature_vector_next = feature_vector_next.view(feature_vector_next.size(0),-1)
		
		next_state_image = torch.cat((next_state_,feature_vector_next),dim=1)
		# get argmax_a' from the CEM for the target Q(s', a'), together with updating the CEM stored weights
		# print(next_state_.shape)
		new_next_action = []
		for i in range(batch_size):      # batch of states, use them one by one
			new_next_action.append(self.cem_optimal_action(next_state_[i],next_state_image[i], image_next[i]))

		new_next_action=torch.FloatTensor(new_next_action).to(self.device)

		target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action, image_next)[0], self.target_qnet2(next_state_, new_next_action, image_next)[0])
		target_q = reward + (1-done)*gamma*target_q_min

		q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
		print('Q Loss: ',q_loss)
		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		# update the target nets, according to original paper:
		# one with Polyak averaging, another with lagged/delayed update
		self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
		self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)

		# ipdb.set_trace()
	

	def cem_optimal_action(self, state, state_image, image_next):
		''' evaluate action wrt Q(s,a) to select the optimal using CEM
		return the only one largest, very gready
		state_image: gripper_states+image_feature vector
		'''
		numpy_state = state_image.detach().numpy()

		''' the following line is critical:
		every time use a new/initialized cem, and cem is only for deriving the argmax_a', 
		but not for storing the weights of the policy.
		Without this line, the Q-network cannot converge, the loss will goes to infinity through time.
		I think the reason is that if you try to use the cem (gaussian distribution of policy weights) fitted 
		to the last state for the next state, it will generate samples mismatched to the global optimum for the 
		current state, which will lead to a local optimum for current state after cem iterations. And there may be
		several different local optimum for a similar state using cem from different last state, which will cause
		the optimal Q-value cannot be learned and even have a divergent loss for Q learning.
		'''
		self.cem.initialize()  # the critical line
		for itr in range(self.cem_update_itr):
			q_values=[]
			theta_list = self.cem.sample_multi(self.num_samples)
			# ipdb.set_trace()
			# print(theta_list)
			for j in range(self.num_samples):
				self.policy.update(theta_list[j])
				one_action = torch.FloatTensor(self.policy.act(numpy_state)).to(self.device)
				# print(one_action)
				q_value,_ = self.target_qnet1(state.unsqueeze(0), one_action,image_next.unsqueeze(0))
				# ipdb.set_trace()
				q_values.append(q_value.detach().cpu().numpy()[0][0]) # 2 dim to scalar
			idx=np.array(q_values).argsort()[-int(self.select_num):]  # select maximum q
			max_idx=np.array(q_values).argsort()[-1]  # select maximal one q
			selected_theta = theta_list[idx]
			mean, _= self.cem.update(selected_theta)  # mean as the theta for argmax_a'
			self.policy.update(mean)
		max_theta=theta_list[max_idx]
		self.policy.update(max_theta)
		action = self.policy.act(numpy_state)[0] # [0]: 2 dim -> 1 dim
		# print("Action: ",action.shape)
		return action

	def target_soft_update(self, net, target_net, soft_tau):
		''' Soft update the target net '''
		for target_param, param in zip(target_net.parameters(), net.parameters()):
			target_param.data.copy_(  # copy data value into target parameters
				target_param.data * (1.0 - soft_tau) + param.data * soft_tau
			)

		return target_net

	def target_delayed_update(self, net, target_net, update_delay):
		''' delayed update the target net '''
		if self.step_cnt%update_delay == 0:
			for target_param, param in zip(target_net.parameters(), net.parameters()):
				target_param.data.copy_(  # copy data value into target parameters
					param.data 
				)

		return target_net

	def save_model(self, path):
		torch.save(self.qnet.state_dict(), path)
		torch.save(self.target_qnet1.state_dict(), path)
		torch.save(self.target_qnet2.state_dict(), path)

	def load_model(self, path):
		self.qnet.load_state_dict(torch.load(path))
		self.target_qnet1.load_state_dict(torch.load(path))
		self.target_qnet2.load_state_dict(torch.load(path))
		self.qnet.eval()
		self.target_qnet1.eval()
		self.target_qnet2.eval()