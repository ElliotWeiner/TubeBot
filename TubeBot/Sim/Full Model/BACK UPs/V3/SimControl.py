import rospy
from std_msgs.msg import UInt8MultiArray, Float32MultiArray, Float32, UInt8, Bool

###########################
# Settings
###########################
# node refresh rate
REFRESH_RATE = 10

# global msg objects, with default values
joint_msg = Float32MultiArray()
joint_msg.data = [0] * 5

suction_msg = UInt8MultiArray()
suction_msg.data = [0] * 2
    
###########################
# Callback Functions
###########################
def callback_end(data):
    if  not data.data:
        print("Node Shutting Down, control_node")
        rospy.signal_shutdown("control stoped")

def cb_r_leg(data):
    joint_msg.data[4] = data.data
    
def cb_r_elbow(data):
    joint_msg.data[3] = data.data
    
def cb_body(data):
    joint_msg.data[2] = data.data
    
def cb_l_elbow(data):
    joint_msg.data[1] = data.data
    
def cb_l_leg(data):
    joint_msg.data[0] = data.data
    
def cb_r_suction(data):
    if data.data:
        suction_msg.data[1] = 1
    else:
        suction_msg.data[1] = 0
    
def cb_l_suction(data):
    if data.data:
        suction_msg.data[0] = 1
    else:
        suction_msg.data[0] = 0
        
def cb_camera_pos(data):
    pass

###########################
# Main Code
###########################
try:
    print("Node Starting, sim control node")
    
    rospy.init_node('SimControl', anonymous=False) # enforces the unique name
    r = rospy.Rate(REFRESH_RATE)
    
    # topic creation
    rospy.Subscriber("SimRunning", Bool, callback_end)
     
    joints = rospy.Publisher("/sim_joints", Float32MultiArray, queue_size=10)
    suctions = rospy.Publisher("/sim_suction", UInt8MultiArray, queue_size=10)
    
    rospy.Subscriber("/right_leg", Float32, cb_r_leg)
    rospy.Subscriber("/right_elbow", Float32, cb_r_elbow)
    rospy.Subscriber("/body", Float32, cb_body)
    rospy.Subscriber("/left_elbow", Float32, cb_l_elbow)
    rospy.Subscriber("/left_leg", Float32, cb_l_leg)
    
    rospy.Subscriber("/right_suction", Bool, cb_r_suction)
    rospy.Subscriber("/left_suction", Bool, cb_l_suction)
    
    rospy.Subscriber("/camera_pos", UInt8, cb_camera_pos)

    while not rospy.is_shutdown():       
        joints.publish(joint_msg)
        suctions.publish(suction_msg)

        r.sleep()
        

except rospy.ROSInterruptException:
    print("Node failed to Start")
