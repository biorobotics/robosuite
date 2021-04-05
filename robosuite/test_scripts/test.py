import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt

is_render = True
# create environment instance
env = suite.make(
    env_name="PickPlaceSingle", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=is_render,
    has_offscreen_renderer=not is_render,
    use_camera_obs=not is_render,
    render_camera='birdview',
    camera_names = 'birdview'              # visualize the "frontview" camera
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    plt.show()
    if is_render:
    	env.render()  # render on display
    else:
    	plt.imshow(obs['image-state'])
    	print("action: ",action)