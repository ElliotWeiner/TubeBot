# ROS related
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

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


# global variables
right_depth = []
right_rgb = []
left_depth = []
left_rgb = []

right_location = [0,0,0]
left_location = [0,0,0]

bridge = CvBridge()

# fucntion for storing depth data
def cb_ight_depth(data):
    global right_depth, bridge
    right_depth = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# function for storing RGB data
def cb_ight_rgb(data):
    global right_rgb, bridge
    right_rgb = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# fucntion for storing depth data
def cb_left_depth(data):
    global left_depth, bridge
    left_depth = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
# function for storing RGB data
def cb_left_rgb(data):
    global left_rgb, bridge
    left_rgb = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
    
def cb_right_pos(data):
    global right_location

    right_location[0] = data.x
    right_location[1] = data.y
    right_location[2] = data.z

def cb_left_pos(data):
    global left_location

    left_location[0] = data.x
    left_location[1] = data.y
    left_location[2] = data.z

 # function to automatically handle simulation stopping
def callback_end(data):
    if  not data.data:
        print("Node Shutting Down, imagenode")
        rospy.signal_shutdown("control stoped")

###########################
# Main Code
###########################
try:
    # start up message
    print("Node Starting, image_node")
    
    # node creation
    rospy.init_node('ImageTap', anonymous=False) # enforces the unique name
    
    # topic creation
    rospy.Subscriber("/SimRunning", Bool, callback_end)
    
    rospy.Subscriber("/right_Depth_image", Image, cb_ight_depth)
    rospy.Subscriber("/right_RGB_image", Image, cb_ight_rgb)
    rospy.Subscriber("/left_Depth_image", Image, cb_left_depth)
    rospy.Subscriber("/left_RGB_image", Image, cb_left_rgb)
    
    rospy.Subscriber("/right_ref", Point, cb_right_pos)
    rospy.Subscriber("/left_ref", Point, cb_left_pos)
    
    # timing set
    r = rospy.Rate(REFRESH_RATE)
    
    ###########################
    # Single Time Setup
    ###########################
    # sets matplotlib into interactive mode and removes tick marks
    plt.ion()
    _, axes = plt.subplots(2, 2)
    
    axis0 = axes[0][0]
    axis1 = axes[0][1]
    axis2 = axes[1][0]
    axis3 = axes[1][1]
    
    # remove tick marks
    for axis in axes.flatten():
        axis.tick_params(left = False, right = False , labelleft = False ,  labelbottom = False, bottom = False) 
        
    # reduce whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.gcf().set_figheight(6)
    
    while not rospy.is_shutdown():       
        
        # displays the stored images        
        if len(right_rgb) > 0:        
            plt.sca(axis1)
            plt.imshow(right_rgb)
            
        if len(right_depth) > 0:        
            plt.sca(axis3)
            plt.imshow(right_depth)
            
        if len(left_rgb) > 0:        
            plt.sca(axis0)
            plt.imshow(left_rgb)
            
        if len(left_depth) > 0:        
            plt.sca(axis2)
            plt.imshow(left_depth)
            
        # sets the titles to identify right and left as well as the locations
        axis1.set_title(f"Right:\n{right_location[0]: .3f}, {right_location[1]: .3f}, {right_location[2]: .3f}")
        axis0.set_title(f"Left:\n{left_location[0]: .3f}, {left_location[1]: .3f}, {left_location[2]: .3f}")
        
        
        # shows and smooths image display process
        plt.pause(0.01)
        
        r.sleep()

except rospy.ROSInterruptException:
    print("Node failed to Start")
