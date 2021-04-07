import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import transforms3d as t3d
import ipdb

from robosuite import load_controller_config

'''
Frames:

b -> base frame of the robot
i -> iPhone frame
d -> desired farme of the robot

'''

def calculate_ee_ori(obs):

		R_id = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
		q_bi = obs['iPhone_quat']
		R_bi = t3d.quaternions.quat2mat(np.array([q_bi[3],q_bi[0],q_bi[1],q_bi[2]]))
		R_bd = np.matmul(R_bi,R_id)
		ax_bd = t3d.axangles.mat2axangle(R_bd)

		return ax_bd


def goto_initial_position(env):
# reset the environment

	env.reset()
	action = np.zeros(7)
	done = False	
	
	for i in range(150):
		print("Current status: Stage 1 on step {}".format(i))
		# action = np.random.randn(env.robots[0].dof) # sample random action
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		# print(obs.keys())
		object_pos = obs['iPhone_pos']
		object_or = obs['iPhone_quat']


		ax_bd = calculate_ee_ori(obs)

		if i<500:
		# print("Object pos: ",object_pos)
		# print("Object or: ",t3d.quaternions.quat2axangle(object_or))
			# R_bi = t3d.quaternions.quat2mat(object_or)
			# R_br= np.matmul(R_bi,R_ie)

			# ax_r = t3d.axangles.mat2axangle(R_br)


			action[0:3] = object_pos
			action[2] += 0.2	
			# action[3:6] = np.array([-ax_r[0][1]*ax_r[1],-ax_r[0][2]*ax_r[1],ax_r[0][0]*ax_r[1]])
			action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
			# action[3:6] = 0
			# object_or = obs['iPhone_quat']
			# print("actual robot pose: ",t3d.quaternions.quat2axangle(obs['robot0_eef_quat']))
			# print("diff in z: ",obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2])
			# print("given robot pose: ",action[3:6])
			# action[6] = 0
			if is_render:
				env.render()  # render on display
			else:
				plt.imshow(obs['image-state'])
				plt.show()
				print("action: ",action)

		if i==500:

			ipdb.set_trace()

	return obs,ax_bd

def goto_down(env,obs,ax_bd):

	i = 0
	action = np.zeros(7)
	done = False
	while (np.abs(obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2] > 0.007)) and i<1000:

		# ax_bd = calculate_ee_ori(obs)

		# ipdb.set_trace()
		print("Current status: Stage 2 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		action[2] -= 0.003
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		if is_render:
				env.render()  # render on display
		else:
			plt.imshow(obs['image-state'])
			plt.show()
			print("action: ",action)
		i+= 1

	return obs


	pass

def goto_close_gripper(env,obs,ax_bd):

	i = 0
	action = np.zeros(7)
	done = False
	for i in range(100):

		# ipdb.set_trace()
		print("Current status: Stage 3 on step {} ".format(i))
		action[0:3] = obs['robot0_eef_pos']
		# ax_bd = calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		action[6] = 1
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		if is_render:
				env.render()  # render on display
		else:
			plt.imshow(obs['image-state'])
			plt.show()
			print("action: ",action)
		i+= 1

	return obs

def goto_up(env,obs,ax_bd):

	i = 0
	action = np.zeros(7)
	action[3] = 1
	done = False
	while (np.abs(obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2] < 0.1)) and i<1000:

		# ipdb.set_trace()
		print("Current status: Stage 4 on step {} and distance is {}".format(i,obs['robot0_eef_pos'][2]-obs['iPhone_pos'][2]))
		action[0:3] = obs['robot0_eef_pos']
		# ax_bd = calculate_ee_ori(obs)
		action[3:6] = np.array([ax_bd[0][0]*ax_bd[1],ax_bd[0][1]*ax_bd[1],ax_bd[0][2]*ax_bd[1]])
		action[2] += 0.003
		if not done:
			obs, reward, done, info = env.step(action)  # take action in the environment
		if is_render:
				env.render()  # render on display
		else:
			plt.imshow(obs['image-state'])
			plt.show()
			print("action: ",action)
		i+= 1

	return obs


	pass

if __name__ == "__main__":

	is_render = True
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




	obs,ax_bd = goto_initial_position(env)
	obs = goto_down(env,obs,ax_bd)
	obs = goto_close_gripper(env,obs,ax_bd)
	obs = goto_up(env,obs,ax_bd)

	# ipdb.set_trace()
	



