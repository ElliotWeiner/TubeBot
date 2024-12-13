from affordance.affordance import affordance
from Sim.sim_functions import sim

import cv2
import numpy as np
import time


def control(goal_pos):
	'''
	inputs:
		init_point:
		goal_point:

	outputs:
		Completed: 	Boolean
	'''

	# init affordance module
	aff_module = affordance()

	# init simulator
	simulator = sim()

	# init policy
	policy = 
	policy.eval()

	# init threshold for goal radius
	threshold = 0.5



	while(True):
		# set init pose
		simulator.init_pose()

		# get picture
		image = simulator.get_image() # Make Function

		# get affordance
		affordance_map = affordance_step(aff_module, image)

		current_pos = simulator.get_base()

		# get control from policy
		length, theta = policy.forward(affordance_map, current_pos, goal_pos)

		# execute control
		simulator.step(length, theta)

		# exit when done
		distance = simulator.distance_to_goal()
		if distance < threshold:
			break
	
	return True

def affordance_step(module, image):
	aff, _ = module.get_affordance(image, verbose=True)

	return aff

# depricated
def affordance_map(image):	# load the image
	
	
	
	model = affordance()

	# resize image if necessary (remove later)
	image = model.resize(image)
	gray = model.edge_image(image)

	# calculate affordance
	grain, aff_img = model.graininess()
	# we want to map affordance values to points
	# we want to check our final position value and determine an approximate affordance value for it.
	# premapping!

	
	model.make_map()

	model.trim(aff_img)

	'''
	# display

	gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	grain = cv2.cvtColor(grain, cv2.COLOR_GRAY2RGB)
	aff_img = cv2.cvtColor(aff_img, cv2.COLOR_GRAY2RGB)


	display(image, gray, grain, aff_img)
	'''

	return aff_img
	
# depricated
# display function
def display(image1, image2, image3, image4):
    combined = np.hstack((image1, image2, image3, image4))
    cv2.imshow("Stages", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
	control()
