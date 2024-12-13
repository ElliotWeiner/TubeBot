from segmentation.expert import Expert


from affordance.affordance import affordance

import cv2
import numpy as np


def main(image_path):
	affordance_map(image_path)


def affordance_map(image_path):	# load the image
	#c_angle = 0.7854
	#c_height =
	# in radians
	angleX = 0.9948
	angleY = 0.7734
	
	camera_parameters = (angleX, angleY)
	
	
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError("Image not found. Please check the path.")
		
	depth = cv2.imread("data/images/22_28_43_479632_depth.png")
	if depth is None:
		raise ValueError("Image not found. Please check the path.")
		
	

	model = affordance()
	seg = Expert()
	print("start")


	# resize image if necessary (remove later)
	image = model.resize(image)
	depth = model.resize(depth)
	gray = model.edge_image(image)


	# get masks
	# only takes a few seconds with smaller images
	masks = seg.predict(image)
	
	# get a masked image, unecessary later on
	masked = seg.get_image(masks, image)

	# make an image of usable masks
	#pmap = model.get_planes(depth, camera_parameters)

	# calculate affordance
	grain, aff_img = model.graininess()
	
	# convert for visualization
	gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	grain = cv2.cvtColor(grain, cv2.COLOR_GRAY2RGB)
	aff_img = cv2.cvtColor(aff_img, cv2.COLOR_GRAY2RGB)

	display(image, gray, grain, aff_img, masked)


# display function
def display(image1, image2, image3, image4, image5):
    combined = np.hstack((image1, image2, image3, image4, image5))
    cv2.imshow("Original and Bilateral Blur", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main("data/images/22_28_43_479632_rgb.png")

