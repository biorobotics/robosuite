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

#['action', 'image_current', 'image_next', 'reward', 'state_current', 'state_next']
with h5py.File('./data/experience_replay_10.h5', 'r') as f:
	replay_keys = list(f.keys())
	print("keys: ",	replay_keys)
	print('actions: ',f['action'][600,2])
	i = f['image_next'][1200]
	# ipdb.set_trace()
	# cv2.imshow("obs image",i)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

