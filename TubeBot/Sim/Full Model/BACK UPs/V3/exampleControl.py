import rospy
from std_msgs.msg import Float32, UInt8, Bool

###########################
# Main Code
###########################
try:
    print("Node Starting, sim control node")
    
    rospy.init_node('ExampleControl', anonymous=False) # enforces the unique name
    r =  rospy.Rate(1)
    
    # topic creation    
    right_leg = rospy.Publisher("/right_leg", Float32, queue_size=10)
    right_elbow = rospy.Publisher("/right_elbow", Float32, queue_size=10)
    body = rospy.Publisher("/body", Float32, queue_size=10)
    left_elbow = rospy.Publisher("/left_elbow", Float32, queue_size=10)
    left_leg = rospy.Publisher("/left_leg", Float32, queue_size=10)
    
    right_suction = rospy.Publisher("/right_suction", Bool, queue_size=10)
    left_suction = rospy.Publisher("/left_suction", Bool, queue_size=10)
    
    camera_pos = rospy.Publisher("/camera_pos", UInt8, queue_size=10)

    ## Value Set
    while not rospy.is_shutdown():  
        right_leg.publish(0)
        right_elbow.publish(.75)
        body.publish(0)
        left_elbow.publish(0)
        left_leg.publish(0)
        
        right_suction.publish(True)
        left_suction.publish(False)
        
        camera_pos.publish(0)  
   
        r.sleep()
    
        

except rospy.ROSInterruptException:
    print("Node failed to Start")
