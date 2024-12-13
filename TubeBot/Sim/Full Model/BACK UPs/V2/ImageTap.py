# ROS related
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

# image viewing related
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np

# image saving related
from datetime import datetime
import os
from PIL import Image as PIL_Image


# settings
## node refresh rate
REFRESH_RATE = 1

## will end the node when the simulation is stopped
STOP_W_SIM = False

# global variables
right_depth = []
right_rgb = []
left_depth = []
left_rgb = []

bridge = CvBridge()

# fucntion for storing depth data
def callback_right_depth(data):
    global right_depth, bridge
    right_depth = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# function for storing RGB data
def callback_right_rgb(data):
    global right_rgb, bridge
    right_rgb = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# fucntion for storing depth data
def callback_left_depth(data):
    global left_depth, bridge
    left_depth = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# function for storing RGB data
def callback_left_rgb(data):
    global left_rgb, bridge
    left_rgb = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)

 # function to automatically handle simulation stopping
def callback_end(data):
    if  not data.data:
        print("Node Shutting Down, imagenode")
        rospy.signal_shutdown("control stoped")

try:
    # start up message
    print("Node Starting, image_node")
    
    # node creation
    rospy.init_node('ImageTap', anonymous=True)
    
    # subscriber creation
    if STOP_W_SIM:
        rospy.Subscriber("SimRunning", Bool, callback_end)
        
    rospy.Subscriber("right_Depth_image", Image, callback_right_depth)
    rospy.Subscriber("right_RGB_image", Image, callback_right_rgb)
    rospy.Subscriber("left_Depth_image", Image, callback_left_depth)
    rospy.Subscriber("left_RGB_image", Image, callback_left_rgb)
    
    # timing set
    r = rospy.Rate(REFRESH_RATE)
    
    # sets matplotlib into interactive mode and removes tick marks
    plt.ion()
    _, axes = plt.subplots(2, 2)
    axis0 = axes[0][0]
    axis1 = axes[0][1]
    axis2 = axes[1][0]
    axis3 = axes[1][1]
    
    axis0.tick_params(left = False, right = False , labelleft = False ,  labelbottom = False, bottom = False) 
    axis0.set_title("Right RGB")
    
    axis1.tick_params(left = False, right = False , labelleft = False ,  labelbottom = False, bottom = False) 
    axis1.set_title("Right Depth")
    
    axis2.tick_params(left = False, right = False , labelleft = False ,  labelbottom = False, bottom = False) 
    axis2.set_title("Left RGB")
    
    axis3.tick_params(left = False, right = False , labelleft = False ,  labelbottom = False, bottom = False) 
    axis3.set_title("Left Depth")
    
    while not rospy.is_shutdown():       
        
        if len(right_rgb) > 0:        
            plt.sca(axis0)
            plt.imshow(right_rgb)
            
        if len(right_depth) > 0:        
            plt.sca(axis1)
            plt.imshow(right_depth)
            
        if len(left_rgb) > 0:        
            plt.sca(axis2)
            plt.imshow(left_rgb)
            
        if len(left_depth) > 0:        
            plt.sca(axis3)
            plt.imshow(left_depth)
        
        
        # shows and smooths image display process
        plt.pause(0.01)
        
        r.sleep()

except rospy.ROSInterruptException:
    print("Node failed to Start")
