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
'''
Frames:

b -> base frame of the robot
i -> iPhone frame
d -> desired farme of the robot

'''
replay = []
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
	elif obs['robot0_gripper_qpos'][0]>0.4:
		gripper_obs[0] = 0	
	gripper_obs[1] = obs['robot0_eef_pos'][2]	#gripper height

	return gripper_obs
	

def relevant_action(action,obs,done):

	gripper_action = np.zeros(7)
	gripper_action[0:3] = action[0:3] - obs['robot0_eef_pos']	#gripper goal vector				
	gripper_action[3:6] = action[3:6]							#gripper rotation
	gripper_action[6] = 1-action[6]								#Open/close gripper open:1, close:0
							#done condition
	return gripper_action
	pass

def relevant_reward(obs):

	# print("reward: ",obs['iPhone_pos'][2])
	#iPhone_z = 0.82
	if obs['iPhone_pos'][2] > 0.82+0.05:
		reward = 1
	else:
		reward = 0

	return reward
	pass

def preprocess_image(image):

	image = image[::-1,0:125]
	image = cv2.resize(image,(256,256))
	# plt.imshow(image)
	# plt.show()
	return image

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
	
	for i in range(150):
		# print("current gripper angles: ",obs['robot0_gripper_qpos'])
		print("Current status: Stage 1 on step {}".format(i))
		# action = np.random.randn(env.robots[0].dof) # sample random action
		
		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)

		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([256,256])

		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		
		else:
			return obs,ax_bd

		if is_render:
			env.render()  # render on display
			image_next = np.zeros([256,256])
		else:
			image_next = preprocess_image(obs['image-state'])
			
		obs_next = relevant_obs(obs)
		reward_current = relevant_reward(obs)
		replay.append(create_experience_replay(obs_current,action_current,reward_current,obs_next,image_current,image_next,done))

		# ipdb.set_trace()



		object_pos = obs['iPhone_pos']
		object_or = obs['iPhone_quat']


		ax_bd = calculate_ee_ori(obs)

		action[0:3] = object_pos
		action[2] += 0.2	
		# action[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])


	return obs,ax_bd

def goto_down(env,obs,ax_bd):

	global replay
	i = 0
	action = np.zeros(7)
	done = False
	while (np.abs(obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2] > 0.007)) and i<1000:
		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([256,256])

		# ax_bd = calculate_ee_ori(obs)

		# ipdb.set_trace()
		print("Current status: Stage 2 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		action[2] -= 0.003
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])

		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment

		if is_render:
			env.render()  # render on display
			image_next = np.zeros([256,256])
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
			image_current = np.zeros([256,256])

		# ipdb.set_trace()
		print("Current status: Stage 3 on step {} ".format(i))
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
			image_next = np.zeros([256,256])
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
	action[3] = 1
	done = False
	while (np.abs(obs['robot0_eef_pos'][2]-0.82 < 0.1)) and i<1000:

		if not is_render:
			image_current = preprocess_image(obs['image-state'])
		else:
			image_current = np.zeros([256,256])
		# print("current gripper angles: ",obs['robot0_gripper_qpos'])
		# ipdb.set_trace()
		print("Current status: Stage 4 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		# ax_bd = calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		action[2] += 0.003
		action_current = relevant_action(action,obs,done)
		obs_current = relevant_obs(obs)
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		
		if is_render:
			env.render()  # render on display
			image_next = np.zeros([256,256])
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

	for episode_period in range(5):
		obs,ax_bd = goto_initial_position(env)
		obs = goto_down(env,obs,ax_bd)
		obs = goto_close_gripper(env,obs,ax_bd)
		obs = goto_up(env,obs,ax_bd)
		# ipdb.set_trace()
		# print("Len of replay: ",(replay))
		replay = np.array(replay)
		# ipdb.set_trace()
		# print(json_string)

		if not is_render:
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

		# ipdb.set_trace()
	



