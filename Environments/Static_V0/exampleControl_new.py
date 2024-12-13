import rospy
from std_msgs.msg import Float32, UInt8, Bool
from geometry_msgs.msg import Point
import numpy as np
from config_functions import *

###########################
# Global Variables
###########################
REFRESH_RATE = 1

right_location = np.array([0,0,0], dtype="float64")
left_location = np.array([0,0,0], dtype="float64")


###########################
# Callback Functions
###########################
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

###########################
# Main Code
###########################
try:
    print("Node Starting, sim control node")
    
    rospy.init_node('ExampleControl', anonymous=False) # enforces the unique name
    r =  rospy.Rate(REFRESH_RATE)
    
    # topic creation    
    right_leg = rospy.Publisher("/right_leg", Float32, queue_size=10)
    right_elbow = rospy.Publisher("/right_elbow", Float32, queue_size=10)
    body = rospy.Publisher("/body", Float32, queue_size=10)
    left_elbow = rospy.Publisher("/left_elbow", Float32, queue_size=10)
    left_leg = rospy.Publisher("/left_leg", Float32, queue_size=10)
    
    right_suction = rospy.Publisher("/right_suction", Bool, queue_size=10)
    left_suction = rospy.Publisher("/left_suction", Bool, queue_size=10)
    
    rospy.Subscriber("/right_ref", Point, cb_right_pos)
    rospy.Subscriber("/left_ref", Point, cb_left_pos)

    current_config = [0,0,0,0,0,0,1]

    # Initial Config
    right_leg.publish(0)
    right_elbow.publish(0)
    body.publish(0)
    left_elbow.publish(0)
    left_leg.publish(0)
    
    right_suction.publish(1)
    left_suction.publish(0)

    
    # Move into camera position
    path = camera_pose_path(current_config, 1)
    
    for config in path:
        for val in config:
            print(f"{val:0.3}  ", end="")
        print("")
        
        left_suction.publish(int(config[0]))
        left_leg.publish(config[1])
        left_elbow.publish(config[2])
        body.publish(config[3])
        right_elbow.publish(config[4])
        right_leg.publish(config[5])
        right_suction.publish(int(config[6]))
        
        r.sleep()

    current_config = path[-1, :]
        
    # Face goal at (-5, -5)
    path = face_goal_path(current_config, 1, right_location, left_location, [-5, -5, 0])
    
    for config in path:
        for val in config:
            print(f"{val:0.3}  ", end="")
        print("")
        
        left_suction.publish(int(config[0]))
        left_leg.publish(config[1])
        left_elbow.publish(config[2])
        body.publish(config[3])
        right_elbow.publish(config[4])
        right_leg.publish(config[5])
        right_suction.publish(int(config[6]))
        
        r.sleep()
        
    current_config = path[-1, :]
    
    for i in range(0, 6):
        r.sleep()
        
    # Face goal at (0, -5)
    path = face_goal_path(current_config, 1, right_location, left_location, [0, -5, 0])
    
    for config in path:
        for val in config:
            print(f"{val:0.3}  ", end="")
        print("")
        
        left_suction.publish(int(config[0]))
        left_leg.publish(config[1])
        left_elbow.publish(config[2])
        body.publish(config[3])
        right_elbow.publish(config[4])
        right_leg.publish(config[5])
        right_suction.publish(int(config[6]))
        
        r.sleep()

    current_config = path[-1, :]
    
    for i in range(0, 6):
        r.sleep()
    
    path = heading_length_path(current_config, 1, np.deg2rad(45), 0.4)
    
    for config in path:
        for val in config:
            print(f"{val:0.3}  ", end="")
        print("")
        
        left_suction.publish(int(config[0]))
        left_leg.publish(config[1])
        left_elbow.publish(config[2])
        body.publish(config[3])
        right_elbow.publish(config[4])
        right_leg.publish(config[5])
        right_suction.publish(int(config[6]))

    # Ending Monitor Loop
    while not rospy.is_shutdown():       
        
        print(f"Right: {right_location}")
        print(f" Left: {left_location}\n")
   
        r.sleep()
    
        

except rospy.ROSInterruptException:
    print("Node failed to Start")
