import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import transforms3d as t3d
from collections import OrderedDict
import ipdb
import cv2
from robosuite import load_controller_config
import json
import argparse
from model import QNetwork
from model import QT_Opt

import torch

import random

class ReplayBuffer:

	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	
	def push(self, state, action, reward, next_state, image_current, image_next, done):
		# print("One replay: ",one_replay)
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, image_current, image_next, done)
		self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
	
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, image_current, image_next, done = map(np.stack, zip(*batch)) # stack for each element

		''' 
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		return state, action, reward, next_state, image_current, image_next, done
	
	def __len__(self):
		return len(self.buffer)

class PickPlace_env(object):
	def __init__(self,args):

		self.controller_config = load_controller_config(default_controller=args.controller)
		self.controller_config['control_delta'] = False
		self.args = args
		self.env = suite.make(
			env_name= self.args.env, # try with other tasks like "Stack" and "Door"
			robots= self.args.robot,  # try with other robots like "Sawyer" and "Jaco"
			controller_configs= self.controller_config,            
			has_renderer= self.args.is_render,
			has_offscreen_renderer= not self.args.is_render,
			use_camera_obs= not self.args.is_render,
			render_camera= self.args.render_camera,
			camera_names =  self.args.offscreen_camera,  					# visualize the "frontview" camera
		)
		self.device = torch.device('cpu', 0)
		self.ex_replay = OrderedDict()
		self.replay_buffer = ReplayBuffer(1e5)
		self.qnet = QNetwork().to(self.device) # gpu
		self.target_qnet1 = QNetwork().to(self.device)
		self.target_qnet2 = QNetwork().to(self.device)
		self.qt_opt = QT_Opt(self.replay_buffer,self.qnet, self.target_qnet1, self.target_qnet2)


	def create_experience_replay(self,state,action,reward,next_state):


		self.ex_replay['state'] = state
		self.ex_replay['action'] = action
		self.ex_replay['reward'] = reward
		self.ex_replay['next_state'] = next_state

		return self.ex_replay

	def robot_obs2obs(self,obs):

		gripper_obs = np.zeros(2)
		if obs['robot0_gripper_qpos'][0]<0.25:
			gripper_obs[0] = 1	#Open/close gripper open:1, close:0
		elif obs['robot0_gripper_qpos'][0]>0.4:
			gripper_obs[0] = 0	
		gripper_obs[1] = obs['robot0_eef_pos'][2]	#gripper height
		return gripper_obs
		pass

	def action2robot_cmd(self,action,obs):

		robot_cmd = np.zeros(7)
		robot_cmd[0:3] = action[0:3]+obs['robot0_eef_pos'][0:3]
		robot_cmd[3:6] = action[3:6]
		robot_cmd[6] = 1-action[6]

		return robot_cmd

	def robot_reward(self,obs):

	# print("reward: ",obs['iPhone_pos'][2])
	#iPhone_z = 0.82
		if obs['iPhone_pos'][2] > 0.82+0.05:
			reward = 1
		else:
			reward = 0

		return reward
		pass

	def calculate_ee_ori(self,obs):

		R_id = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
		q_bi = obs['robot0_eef_quat']
		R_bi = t3d.quaternions.quat2mat(np.array([q_bi[3],q_bi[0],q_bi[1],q_bi[2]]))
		# R_bd = np.matmul(R_bi,R_id)
		ax_bd = t3d.axangles.mat2axangle(R_bi)

		return ax_bd

	def select_action(self,obs):

		action = np.zeros(7)
		action[0:3] = [0,0,0]
		ax_bd = self.calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])

		return action

	def preprocess_image(self,image):

		image = image[::-1,0:125]
		image = cv2.resize(image,(256,256))
		return image

	def run(self):

		for i in range(self.args.max_episodes):

			print("Current episode: ",i)
			obs = self.env.reset()
			action = np.zeros(7)
			done = False

			if not self.args.is_render:
				image_current = self.preprocess_image(obs['image-state'])
			else:
				image_current = np.zeros([256,256])

			max_steps = 500

			for s in range(max_steps):

				if not done:
					# ipdb.set_trace()
					obs_current = self.robot_obs2obs(obs)
					obs_     = torch.FloatTensor(obs_current).to(self.device)
					image_current_ = torch.FloatTensor(image_current).to(self.device)
					action_ = torch.FloatTensor(action).to(self.device)

					_, feature_vector_next = self.qnet(obs_.unsqueeze(0), action_.unsqueeze(0), image_current_.unsqueeze(0))
					feature_vector_next = feature_vector_next.view(feature_vector_next.size(0),-1)
					obs_image_ = torch.cat((obs_.unsqueeze(0),feature_vector_next),dim=1)

					action = self.qt_opt.cem_optimal_action(obs_,obs_image_,image_current_)
					# print("action 1: ",action)
					# action = self.select_action(obs)
					# print("action 2: ",action)
					robot_cmd = self.action2robot_cmd(action,obs)

					if self.args.is_render:
						self.env.render()  # render on display
						image = np.zeros([256,256])
					else:
						image_current = self.preprocess_image(obs['image-state'])

					print("Taking the step {}".format(s))
					obs, reward, done, info = self.env.step(robot_cmd)
					reward = self.robot_reward(obs)
					# print("Reward: ",reward)
					if self.args.is_render:
						self.env.render()  # render on display
						image = np.zeros([256,256])
					else:
						image_next = self.preprocess_image(obs['image-state'])

					obs_next = self.robot_obs2obs(obs)
					# reward = reward(obs)
					self.replay_buffer.push(obs_current, action, reward, obs_next, image_current, image_next, done)

					# print("Len of replay buffer: ",len(self.replay_buffer))


			if len(self.replay_buffer) > self.args.batch_size:
				self.qt_opt.update(self.args.batch_size)
			# 	qt_opt.save_model(model_path)






if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='QT-Opt training Parameters')
	parser.add_argument('--normalizer', type=bool, default=True, help='use normalizer')
	# parser.add_argument('--log', type=str, default='exp14', help='Log folder to store videos')
	parser.add_argument('--policy_idx', type=int, default=20, help='Index of model to test the environment')
	parser.add_argument('--env', type=str, default='PickPlaceiPhone', help='name of environment')
	parser.add_argument('--robot', type=str, default='UR5e', help='name of robot')
	parser.add_argument('--controller', type=str, default='OSC_POSE', help='name of controller')
	parser.add_argument('--is_render', type=bool, default=False, help='rendering yes/no')
	parser.add_argument('--render_camera', type=str, default='frontview', help='camera used for online rendering')
	parser.add_argument('--offscreen_camera', type=str, default='agentview', help='rcamera used for offline rendering')
	parser.add_argument('--max_episodes', type=int, default=10, help='max episodes for which env runs')
	parser.add_argument('--batch_size', type=int, default=100, help='episodes after which q values are updated')


	args = parser.parse_args()
	# args.log = os.path.join(args.log)
	pick_place = PickPlace_env(args)
	pick_place.run()
	# ars.test()