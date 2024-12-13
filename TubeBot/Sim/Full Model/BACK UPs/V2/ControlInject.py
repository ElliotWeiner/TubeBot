import rospy
from std_msgs.msg import Float32, Bool

import random

sensor = False

MAX_ROTATION = 2
MAX_LINEAR = 5
REFRESH_RATE = 0.5

## will end the node when the simulation is stopped
STOP_W_SIM = False
    
def callback_end(data):
    if  not data.data:
        print("Node Shutting Down, control_node")
        rospy.signal_shutdown("control stoped")

try:
    print("Node Starting, control_node")
    
    rospy.init_node('ControlInject', anonymous=True)
    r = rospy.Rate(REFRESH_RATE)
    
    # subscriber creation
    if STOP_W_SIM:
        rospy.Subscriber("SimRunning", Bool, callback_end)
        
    rightLeg = rospy.Publisher("rightLeg", Float32, queue_size=10)
    rightElbow = rospy.Publisher("rightElbow", Float32, queue_size=10)
    mainBody = rospy.Publisher("mainBody", Float32, queue_size=10)
    leftElbow = rospy.Publisher("leftElbow", Float32, queue_size=10)
    leftLeg = rospy.Publisher("leftLeg", Float32, queue_size=10)
    
    rightSuction= rospy.Publisher("rightSuction", Bool, queue_size=10)
    leftSuction= rospy.Publisher("leftSuction", Bool, queue_size=10)

    while not rospy.is_shutdown():       
        r.sleep()
        
        rightLeg.publish(0)
        rightElbow.publish(0)
        mainBody.publish(0)
        leftElbow.publish(0)
        leftLeg.publish(0)
        
        rightSuction.publish(False)
        leftSuction.publish(False)
        
        modifier = 2
        
        lin_step = modifier*0.01
        ang_step = modifier*0.1
        steps = int(10 /modifier)
        
        print(f"lin: {lin_step}\nang: {ang_step}\nsteps: {steps}\n")
        
        # rightSuction.publish(True)
        leftSuction.publish(True)
        
        for i in range(0, steps):
            r.sleep()
            rightLeg.publish(i*ang_step)
            
        for i in range(0, steps):
            r.sleep()
            rightElbow.publish(i*ang_step)
            
        for i in range(0, steps):
            r.sleep()
            mainBody.publish(i*lin_step)
            
        for i in range(0, steps):
            r.sleep()
            leftElbow.publish(i*ang_step)
            
        for i in range(0, steps):
            r.sleep()
            leftLeg.publish(i*ang_step)

        # leftSuction.publish(True)
        
        leftLeg.publish(0)
        rightLeg.publish(0)
        
        break
        rospy.spin()

except rospy.ROSInterruptException:
    print("Node failed to Start")
