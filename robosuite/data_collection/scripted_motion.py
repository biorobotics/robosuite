import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import transforms3d as t3d
from collections import OrderedDict
import ipdb
import cv2
from robosuite import load_controller_config
import json
import h5py
import math
'''
Frames:

b -> base frame of the robot
i -> iPhone frame
d -> desired farme of the robot

'''

IMAGE_SIZE = (128,128)
replay = []

def calculate_velocity(obs):

	v = np.zeros(3)
	v[0:2] = (-(obs['robot0_eef_pos'][0:2]-obs['iPhone_pos'][0:2])*0.02)/0.001
	# print(v)
	return v
def calculate_ee_ori(obs):

		R_id = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
		q_bi = obs['iPhone_quat']
		R_bi = t3d.quaternions.quat2mat(np.array([q_bi[3],q_bi[0],q_bi[1],q_bi[2]]))
		R_bd = np.matmul(R_bi,R_id)
		ax_bd = t3d.axangles.mat2axangle(R_bd)

		return ax_bd


def relevant_obs(obs):

	gripper_obs = np.zeros(2)
	if obs['robot0_gripper_qpos'][0]<0.25:
		gripper_obs[0] = 1	#Open/close gripper open:1, close:0
	else:
		gripper_obs[0] = 0	
	gripper_obs[1] = obs['robot0_eef_pos'][2]	#gripper height
	# print("gripper_obs: ",gripper_obs)
	return gripper_obs
	

def relevant_action(action,obs,done):

	gripper_action = np.zeros(5)
	dt = 0.001
	gripper_action[0:3] = (action[0:3] - obs['robot0_eef_pos'])/dt	#gripper goal vector				
	gripper_action[3] = math.atan2(action[4],action[3])*2							#gripper rotation
	gripper_action[4] = 1-action[6] #Open/close gripper open:1, close:0
	# print("gripper_action: ",gripper_action)
	# ipdb.set_trace()
	return gripper_action
	pass

def relevant_reward(obs):

	# print("reward: ",obs['iPhone_pos'][2])
	#iPhone_z = 0.82
	if obs['iPhone_pos'][2] > 0.82+0.02:
		reward_grasp = 1
	else:
		reward_grasp = 0

	reward_pos = np.exp(-np.linalg.norm(obs['iPhone_pos'][0:2]-obs['robot0_eef_pos'][0:2]))
	l1 = 1
	reward_total = reward_grasp+l1*reward_pos

	# print("Reward_total: ",reward_total)
	return reward_total
	pass

def preprocess_image(image):


	image_ = image[::-1,0:125].astype('uint8')
	image_ = cv2.resize(image_,IMAGE_SIZE)
	image_ = cv2.cvtColor(image_,cv2.COLOR_BGR2GRAY)
	image_ = image_.reshape(IMAGE_SIZE[0],IMAGE_SIZE[1],1)
	# cv2.imshow("obs image",image_)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows() 
	return image_

def create_experience_replay(state,action,reward,next_state,image_current,image_next,done):

	# ex_replay = OrderedDict()

	# ex_replay['state'] = state
	# ex_replay['action'] = action
	# ex_replay['reward'] = reward
	# ex_replay['next_state'] = next_state
	# ex_replay['image_current'] = image_current
	# ex_replay['image_next'] = image_next
	reward = np.array(reward).astype('float64')
	done = np.array(done).astype('bool')

	ex_replay = np.array([state,action,reward,next_state,image_current,image_next,done])
	# print("state dtype: ",state.dtype)
	# print("action dtype: ",action.dtype)
	# print("reward: ",reward.dtype)
	# print("image dtype: ",image_current.dtype)
	# print("ex_replay dtype: ",ex_replay.dtype)



	return ex_replay


def goto_initial_position(env):
# reset the environment
	
	global replay
	obs = env.reset()

	# print(obs)
	action = np.zeros(7)
	done = False
	theta = np.random.uniform(low=0, high=3.14)
	while (np.linalg.norm(obs['robot0_eef_pos'][0:2]-obs['iPhone_pos'][0:2]))>0.025:
		# print("termination condition: ",np.linalg.norm(obs['robot0_eef_pos'][0:2]-obs['iPhone_pos'][0:2]))
		# print("current gripper angles: ",obs['robot0_gripper_qpos'])
		# print("Current status: Stage 1 on step {}".format(i))
		# action = np.random.randn(env.robots[0].dof) # sample random action
		v = calculate_velocity(obs)
		# print("v is: ",v.shape)
		action[0:3] = v*0.001 + obs['robot0_eef_pos'][0:3]
		action_current = relevant_action(action,obs,done)
		# print("action_current: ",action_current)
		obs_current = relevant_obs(obs)

		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])

		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		
		else:
			return obs,ax_bd

		if is_render:
			env.render()  # render on display
			image_next= np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])
		else:
			image_next = preprocess_image(obs['image-state'])
			
		obs_next = relevant_obs(obs)
		reward_current = relevant_reward(obs)

			#introduced to remove initial jerky motion of the arm
		replay.append(create_experience_replay(obs_current,action_current,reward_current,obs_next,image_current,image_next,done))

		# ipdb.set_trace()



		object_pos = obs['iPhone_pos']
		object_or = obs['iPhone_quat']


		ax_bd = calculate_ee_ori(obs)

		# action[0:3] = object_pos
		action[2] = object_pos[2] + 0.2	
		# action[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		# action[3:6] = np.array([math.cos(theta/2)*3.14,math.sin(theta/2)*3.14,0])
		# print("Action :",action[3:6])


	return obs,ax_bd

def goto_down(env,obs,ax_bd):

	global replay
	i = 0
	action = np.zeros(7)
	done = False
	while (np.abs(obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2] > 0.007)) and i<2000:
		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])

		# ax_bd = calculate_ee_ori(obs)

		# ipdb.set_trace()
		# print("Current status: Stage 2 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		action[2] -= 0.003
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])

		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment

		if is_render:
			env.render()  # render on display
			image_next = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])
		else:
			image_next = preprocess_image(obs['image-state'])

		obs_next = relevant_obs(obs)
		reward_current = relevant_reward(obs)
		replay.append(create_experience_replay(obs_current,action_current,reward_current,obs_next,image_current,image_next,done))
		i+= 1

	return obs


	pass

def goto_close_gripper(env,obs,ax_bd):

	global replay
	i = 0
	action = np.zeros(7)
	done = False

	for i in range(100):

		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])

		# ipdb.set_trace()
		# print("Current status: Stage 3 on step {} ".format(i))
		action[0:3] = obs['robot0_eef_pos']
		# ax_bd = calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		action[6] = 1
		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment

		if is_render:
			env.render()  # render on display
			image_next = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])
		else:
			image_next = preprocess_image(obs['image-state'])

		obs_next = relevant_obs(obs)
		reward_current = relevant_reward(obs)
		replay.append(create_experience_replay(obs_current,action_current,reward_current,obs_next,image_current,image_next,done))


		i+= 1

	return obs

def goto_up(env,obs,ax_bd):

	global replay
	i = 0
	action = np.zeros(7)
	# action[3] = 1
	done = False
	action[6] = 1
	while (np.abs(obs['robot0_eef_pos'][2]-0.82 < 0.1)) and i<1000:

		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])
		# print("current gripper angles: ",obs['robot0_gripper_qpos'])
		# ipdb.set_trace()
		# print("Current status: Stage 4 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		# ax_bd = calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		action[2] += 0.001
		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		
		if is_render:
			env.render()  # render on display
			image_next = np.zeros([IMAGE_SIZE[0],IMAGE_SIZE[1]])
		else:
			image_next = preprocess_image(obs['image-state'])

		obs_next = relevant_obs(obs)
		reward_current = relevant_reward(obs)
		replay.append(create_experience_replay(obs_current,action_current,reward_current,obs_next,image_current,image_next,done))
		i+= 1


	return obs


	pass

if __name__ == "__main__":

	# global replay
	is_render = False
	controller_names = ["OSC_POSE","OSC_POSITION"]
	controller_config = load_controller_config(default_controller=controller_names[0])
	controller_config['control_delta'] = False
	controller_config['input_max'] = 10
	controller_config['input_min'] = -10

	print("controller_config: ",controller_config)


	collect_data = not is_render
	# create environment instance
	env = suite.make(
		env_name="PickPlaceiPhone", # try with other tasks like "Stack" and "Door"
		robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
		controller_configs=controller_config,            
		has_renderer=is_render,
		has_offscreen_renderer=not is_render,
		use_camera_obs=not is_render,
		render_camera='frontview',
		camera_names = 'agentview',  					# visualize the "frontview" camera
	)

	import h5py

	for episode_period in range(50):
		obs,ax_bd = goto_initial_position(env)
		# print("Obs: ",calculate_ee_ori(obs))
		obs = goto_down(env,obs,ax_bd)
		# print("Obs: ",calculate_ee_ori(obs))
		obs = goto_close_gripper(env,obs,ax_bd)
		# print("Obs: ",calculate_ee_ori(obs))
		obs = goto_up(env,obs,ax_bd)
		# print("Obs: ",calculate_ee_ori(obs))
		
		# ipdb.set_trace()
		# print("Len of replay: ",(replay))
		replay = np.array(replay)
		# ipdb.set_trace()
		# print(json_string)
		if collect_data:
			#obs_current,action_current,reward_current,obs_next,image_current,image_next
			with h5py.File('./data/experience_replay_{}.h5'.format(episode_period+1), "w") as f:

				state_current_ = np.array(list(replay[:, 0]), dtype=np.float)
				dset = f.create_dataset("state", data = state_current_)

				action_ = np.array(list(replay[:, 1]), dtype=np.float)
				dset = f.create_dataset("action", data = action_ )

				reward_ = np.array(list(replay[:, 2]), dtype=np.uint8)
				dset = f.create_dataset("reward", data = reward_)

				state_next_ = np.array(list(replay[:, 3]), dtype=np.float)
				dset = f.create_dataset("next_state", data = state_next_ )

				image_current_ = np.array(list(replay[:, 4]), dtype=np.uint8)
				dset = f.create_dataset("image_current", data = image_current_)

				image_next_ = np.array(list(replay[:, 5]), dtype=np.uint8)
				dset = f.create_dataset("image_next", data = image_next_)

				done_ = np.array(list(replay[:, 6]), dtype=np.bool)
				dset = f.create_dataset("done", data = done_)

			
		replay = []
		print("Collected replay for episode {}".format(episode_period+1))

		# ipdb.set_trace()
	



