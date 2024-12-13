from enum import IntEnum
import numpy as np

import rospy
import ros_numpy
from std_msgs.msg import UInt8MultiArray, Float32MultiArray
from geometry_msgs.msg import Point


def get_angle(p1, p2):
    """
        gets the angle from p1 to p2, bounded in [0, 2pi)
    """
    
    p1 = np.array(p1).flatten()
    p2 = np.array(p2).flatten()
    
    if p1.size > 2:
        p1 = p1[[0,1]]
        
    if p2.size > 2:
        p2 = p2[[0,1]]
    
    p_diff = p2-p1
    
    if p_diff[0] == 0:
        if p_diff[1] > 0:
            angle = np.pi/2
            
        elif p_diff[1] < 0:
            angle = -np.pi/2
            
        else:
            angle = 0
    else:
        angle = np.arctan(p_diff[1] / p_diff[0])
    
    if p_diff[0] < 0:
        angle = np.pi + angle
        
    if angle < 0:
        angle += 2*np.pi
    
    return angle
    
def to_FloatMsg(data):
    msg = Float32MultiArray()
    msg.data = data.tolist()
    return msg
    
def to_IntMsg(data, is_np=True):
    msg = UInt8MultiArray()
    if is_np:
        msg.data = data.astype(int).tolist()
    else:
        msg.data = data
    return msg

class R_VAR(IntEnum):
    L_SUCTION = 0
    L_LEG = 1
    L_ELBOW = 2
    BODY = 3
    R_ELBOW = 4
    R_LEG = 5
    R_SUCTION = 6
    
    def joints():
        return [1,2,3,4,5]
        
    def suctions():
        return [0,6]

class Virtual_Robot:
    def __init__(self, rate=10, steps=10):
        self.steps = steps
        
        self.curr_pos = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0])
        self.next_pos = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0])
        
        self.left_foot = np.array([0.0, 0.0, 0.0])
        self.right_foot = np.array([0.0, 0.0, 0.0])
        
        def cb_position_update(position, data):
            position[0] = np.round(data.x, 8)
            position[1] = np.round(data.y, 8)
            position[2] = np.round(data.z, 8)
        
        try:
            print("VirtualRobot Node Starting")
            rospy.init_node("VirtualRobot", anonymous=False) # enforces the unique name
            self.ros_timer = rospy.Rate(rate)
            
            self.joints = rospy.Publisher("/sim_joints", Float32MultiArray, queue_size=10)
            self.suctions = rospy.Publisher("/sim_suctions", UInt8MultiArray, queue_size=10)
            
            rospy.Subscriber("/left_ref", Point, lambda data: cb_position_update(self.left_foot, data))
            rospy.Subscriber("/right_ref", Point, lambda data: cb_position_update(self.right_foot, data))
            
            self.__push__()
            
        except rospy.ROSInterruptException:
            print("VirtualRobot failed to start")

    def __str__(self):
        return f"{self.curr_pos.tolist()}\n{self.next_pos.tolist()}\n{self.left_foot.tolist()}\n{self.right_foot.tolist()}"

    def __push__(self):
        for i in range(0, 3):
            self.joints.publish(to_FloatMsg(self.curr_pos[R_VAR.joints()]))
            self.suctions.publish(to_IntMsg(self.curr_pos[R_VAR.suctions()]))
            
            self.ros_timer.sleep()

    def shutdown(self):
        rospy.signal_shutdown("VirtualRObot Stopping")
        
    def wait(self, delay):
        rospy.sleep(delay)

    def update(self):
        """
            updates the internal start the sim state from curr_pos to next_pos
        """
        
        # Rounds positions
        for i in R_VAR.joints():
            self.curr_pos[i] = np.round(self.curr_pos[i], 4)
            self.next_pos[i] = np.round(self.next_pos[i], 4)
        
        #
        #  Suction Actuation
        #
        # checks if either suction cup changed
        if np.any(self.curr_pos[R_VAR.suctions()] != self.next_pos[R_VAR.suctions()]):
            
            # sets both suctions
            self.curr_pos[R_VAR.suctions()] = np.array([1, 1])
            self.__push__()
            
            # updates current position
            self.curr_pos[R_VAR.suctions()] = self.next_pos[R_VAR.suctions()]
            
            # set the suctions to the desired configuration
            self.__push__()
        
        
        # the robot should only move if 1 or less suctions are active
        if sum(self.curr_pos[R_VAR.suctions()].astype(bool)) <= 1:

            if self.curr_pos[R_VAR.L_SUCTION] == 1 and self.curr_pos[R_VAR.L_LEG] < self.curr_pos[R_VAR.L_LEG]: # left leg is suctioned

                # checks if the left elbow is moving
                if self.curr_pos[R_VAR.L_ELBOW] != self.next_pos[R_VAR.L_ELBOW]:
                    path = np.linspace(self.curr_pos[R_VAR.L_ELBOW], self.next_pos[R_VAR.L_ELBOW], self.steps)
                    
                    for i in range(1,len(path)):
                        self.curr_pos[R_VAR.L_ELBOW] = path[i]
                        self.__push__()
                        a
            elif self.curr_pos[R_VAR.R_SUCTION] == 1 and self.curr_pos[R_VAR.R_LEG] < self.curr_pos[R_VAR.R_LEG]: # right leg is suctioned

                # checks if the right elbow is moving
                if self.curr_pos[R_VAR.R_ELBOW] != self.next_pos[R_VAR.R_ELBOW]:
                    path = np.linspace(self.curr_pos[R_VAR.R_ELBOW], self.next_pos[R_VAR.R_ELBOW], self.steps)
                    
                    for i in range(1,len(path)):
                        # update internal and then update sim
                        self.curr_pos[R_VAR.R_ELBOW] = path[i]
                        self.__push__()
                        
            # moves any remaining joints,
            #  if neither leg was suctioned, then it moves all 5 joints together
            
            # checks the number of joints that differ
            changes = sum(self.curr_pos[R_VAR.joints()] != self.next_pos[R_VAR.joints()])
            if changes > 0:
                path = np.linspace(self.curr_pos[R_VAR.joints()], self.next_pos[R_VAR.joints()], self.steps*changes)

                for i in range(1,len(path)):
                    # update internal and then update sim
                    self.curr_pos[R_VAR.joints()] = path[i]
                    self.__push__()
                    
        # all joints have been moved, if possible
        #  next_pos is updated to match curr_pos
        self.next_pos = np.copy(self.curr_pos)
        
    def set_pose(self, l_suction=None, l_leg=None, l_elbow=None, body=None, r_elbow=None, r_leg=None, r_suction=None):
        """
            queues an arbitrary next pose into the next_pos
        """
        
        if l_suction is not None:
            self.next_pos[R_VAR.L_SUCTION] = int(l_suction)
            
        if l_leg is not None:
            self.next_pos[R_VAR.L_LEG] = l_leg
            
        if l_elbow is not None:
            self.next_pos[R_VAR.L_ELBOW] = l_elbow
            
        if body is not None:
            self.next_pos[R_VAR.BODY] = body
            
        if r_elbow is not None:
            self.next_pos[R_VAR.R_ELBOW] = r_elbow
            
        if r_leg is not None:
            self.next_pos[R_VAR.R_LEG] = r_leg
            
        if r_suction is not None:
            self.next_pos[R_VAR.R_SUCTION] = int(r_suction)
     

    def set_suction(self, left=False, right=False, toggle=True):
        """
            changes queued suction state, by defualt it will toggle the left & right states 
                from the current position, if either left or right is set to true, 
                then the toggle will be ignored
        """
        
        if left or right:
            self.next_pos[R_VAR.suctions()] = [int(left), int(right)]
            
        elif toggle:
            self.next_pos[R_VAR.suctions()] = np.logical_not(self.curr_pos[R_VAR.suctions()])
        

    def camera_pose(self, camera_angle=45):
        """
            the next position will be set to go to camera pose based on which foot is suctioned. 
                If both or neither are suctioned, nothing will happend
        """
        
        # checks if only 1 foot is mounted
        if sum(self.curr_pos[R_VAR.suctions()].astype(bool)) == 1:          
            if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
                
                self.next_pos[R_VAR.L_ELBOW] = np.pi/4
                self.next_pos[R_VAR.BODY] = 0
                self.next_pos[R_VAR.R_ELBOW] = 0
                self.next_pos[R_VAR.R_LEG] = 0
                
                
            else: # right foot suctioned
                
                self.next_pos[R_VAR.L_LEG] = 0
                self.next_pos[R_VAR.L_ELBOW] = 0
                self.next_pos[R_VAR.BODY] = 0
                self.next_pos[R_VAR.R_ELBOW] = np.pi/4
 
    def face_point(self, point):
        """
            rotates the robot to face a given point. exactly one foot must be suctioned. 
        """

        # checks if only 1 foot is mounted
        if sum(self.curr_pos[R_VAR.suctions()].astype(bool)) == 1:    
            # converts and flattens the given point
            point = np.array(point).flatten()
            
            if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
                
                curr_heading = get_angle(self.left_foot, self.right_foot)
                desired_heading = get_angle(self.left_foot, point)
                
                self.next_pos[R_VAR.L_LEG] = self.next_pos[R_VAR.L_LEG] + desired_heading - curr_heading
                self.next_pos[R_VAR.L_LEG] = (self.next_pos[R_VAR.L_LEG] + 2*np.pi) % (2*np.pi)
                
            else: # right foot suctioned
                
                curr_heading = get_angle(self.right_foot, self.left_foot)
                desired_heading = get_angle(self.right_foot, point)
                

                self.next_pos[R_VAR.R_LEG] = self.next_pos[R_VAR.R_LEG] + desired_heading - curr_heading
                self.next_pos[R_VAR.R_LEG] = (self.next_pos[R_VAR.R_LEG] + (2*np.pi)) % (2*np.pi)


                    
    def heading_length(self, length, heading):
        """
            rotates & extends the robot to move to a given length and heading. exactly one foot must be suctioned. 
        """

        # checks if only 1 foot is mounted
        if sum(self.curr_pos[R_VAR.suctions()].astype(bool)) == 1:    
            
            if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
                
                curr_heading = get_angle(self.left_foot, self.right_foot)
                
                self.next_pos[R_VAR.L_LEG] = self.next_pos[R_VAR.L_LEG] + heading - curr_heading
                self.next_pos[R_VAR.L_LEG] = (self.next_pos[R_VAR.L_LEG] + 2*np.pi) % (2*np.pi)
                self.next_pos[R_VAR.L_ELBOW] = 0
                self.next_pos[R_VAR.BODY] = length - 0.374323 # offset for min robot length
                self.next_pos[R_VAR.R_ELBOW] = 0
                
                
            else: # right foot suctioned
                
                curr_heading = get_angle(self.right_foot, self.left_foot)
                
                self.next_pos[R_VAR.L_ELBOW] = 0
                self.next_pos[R_VAR.BODY] = length - 0.374323 # offset for min robot length
                self.next_pos[R_VAR.R_ELBOW] = 0
                
                self.next_pos[R_VAR.R_LEG] = self.next_pos[R_VAR.R_LEG] + heading - curr_heading
                self.next_pos[R_VAR.R_LEG] = (self.next_pos[R_VAR.R_LEG] + 2*np.pi) % (2*np.pi)
        

            