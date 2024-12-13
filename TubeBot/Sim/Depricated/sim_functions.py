import rospy
from std_msgs.msg import Float32, UInt8, Bool

import numpy as np
import random
from enum import IntEnum

import rospy
import ros_numpy
from std_msgs.msg import UInt8MultiArray, Float32MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


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

class sim():
    def __init__(self, rate=20, step_per=30, verbose=False):
        self.step_per = step_per
        self.verbose = verbose
        
        self.curr_pos = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0 ,1])
        self.next_pos = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0 ,1])
        
        self.left_foot = np.array([0.0, 0.0, 0.0])
        self.right_foot = np.array([0.0, 0.0, 0.0])
        
        self.left_image = []
        self.right_image = []
        
        self.goal = None
        
        def cb_position_update(position, data):
            position[0] = np.round(data.x, 8)
            position[1] = np.round(data.y, 8)
            position[2] = np.round(data.z, 8)
            
        def cb_image_update(self, left, data):
            bridge = CvBridge()
            if left:
                self.left_image = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
            else:
                self.right_image = np.flip(bridge.imgmsg_to_cv2(data, "rgb8"), 0)
                
                
        def callback_end(data):
            if  not data.data:
                print("Node Shutting Down, control_node")
                rospy.signal_shutdown("control stoped")
        
        
        
        try:
            print("VirtualRobot Node Starting")
            rospy.init_node("VirtualRobot", anonymous=False) # enforces the unique name
            self.ros_timer = rospy.Rate(rate)
            
            self.pub_joints = rospy.Publisher("/sim_joints", Float32MultiArray, queue_size=10)
            self.pub_suctions = rospy.Publisher("/sim_suctions", UInt8MultiArray, queue_size=10)
            
            rospy.Subscriber("/left_ref", Point, lambda data: cb_position_update(self.left_foot, data))
            rospy.Subscriber("/right_ref", Point, lambda data: cb_position_update(self.right_foot, data))
            rospy.Subscriber("/right_RGB_image", Image, lambda data: cb_image_update(self, False, data))
            rospy.Subscriber("/left_RGB_image", Image, lambda data: cb_image_update(self, True, data))
            
            rospy.Subscriber("SimRunning", Bool, callback_end)
            
            self.__push__()
            
        except rospy.ROSInterruptException:
            print("VirtualRobot failed to start")
        
    def __str__(self):
        
        output = []
        output.append(f"steps: {self.step_per}")
        output.append(f"goal: {'None' if self.goal is None else self.goal.tolist()}\n")
        output.append(f"curr: {self.curr_pos.tolist()}")
        output.append(f"next: {self.next_pos.tolist()}\n")
        output.append(f"left foot: {self.left_foot.tolist()}")
        output.append(f"right foot: {self.right_foot.tolist()}\n")
        output.append(f"left image: {np.size(self.left_image) != 0}")
        output.append(f"right image: {np.size(self.right_image) != 0}")

        return "\n".join(output)
        
    def __push__(self, delay=3):
        if self.verbose:
            output=["push:"]
            output.append(f"r:{np.rad2deg(self.curr_pos[[1,2,4,5]]).tolist()}")
            output.append(f"p:{self.curr_pos[3].tolist()}")
            output.append(f"s:{self.curr_pos[R_VAR.suctions()].tolist()}")
            print("  ".join(output))
        
        joint_transform = np.array([-1,1,1,1,-1])
        
        for i in range(0, 3):
            self.pub_suctions.publish(to_IntMsg(self.curr_pos[R_VAR.suctions()]))
            self.ros_timer.sleep()
            
            self.pub_joints.publish(to_FloatMsg(self.curr_pos[R_VAR.joints()] * joint_transform))
            self.ros_timer.sleep()

        for i in range(0, delay):
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
                    path = np.linspace(self.curr_pos[R_VAR.L_ELBOW], self.next_pos[R_VAR.L_ELBOW], self.step_per)
                    
                    for i in range(1,len(path)):
                        self.curr_pos[R_VAR.L_ELBOW] = path[i]
                        self.__push__()
                        a
            elif self.curr_pos[R_VAR.R_SUCTION] == 1 and self.curr_pos[R_VAR.R_LEG] < self.curr_pos[R_VAR.R_LEG]: # right leg is suctioned

                # checks if the right elbow is moving
                if self.curr_pos[R_VAR.R_ELBOW] != self.next_pos[R_VAR.R_ELBOW]:
                    path = np.linspace(self.curr_pos[R_VAR.R_ELBOW], self.next_pos[R_VAR.R_ELBOW], self.step_per)
                    
                    for i in range(1,len(path)):
                        # update internal and then update sim
                        self.curr_pos[R_VAR.R_ELBOW] = path[i]
                        self.__push__()
                        
            # moves any remaining joints,
            #  if neither leg was suctioned, then it moves all 5 joints together
            
            # checks the number of joints that differ
            changes = sum(self.curr_pos[R_VAR.joints()] != self.next_pos[R_VAR.joints()])
            if changes > 0:
                path = np.linspace(self.curr_pos[R_VAR.joints()], self.next_pos[R_VAR.joints()], self.step_per*changes)

                for i in range(1,len(path)):
                    # update internal and then update sim
                    self.curr_pos[R_VAR.joints()] = path[i]
                    self.__push__()
                    
        # all joints have been moved, if possible
        #  next_pos is updated to match curr_pos
        self.next_pos = np.copy(self.curr_pos)
        
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
    
    def face_point(self, point):
        """
            rotates the robot to face a given point. exactly one foot must be suctioned. 
        """            
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
    
    def randomize(self):
        """
            randomizes the position of the robot
        """
        pass
        
    def new_goal(self):
        """
            generates new goal
        """
        new_goal = np.array([random.uniform(-18, 18), random.uniform(-18, 18)])
        self.goal = np.copy(new_goal)
        print(new_goal)
        return new_goal
        
    def get_base(self):
        """
            x,y of suctioned foot
        """
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            return self.left_foot[:-1]
                
        else: # right foot suctioned
            return self.right_foot[:-1]
    
    def init_pose(self, camera_angle=45):
        """
            the next position will be set to go to camera pose based on which foot is suctioned. 
                If both or neither are suctioned, nothing will happend
        """
        
        camera_angle = np.deg2rad(camera_angle)
        
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            
            self.next_pos[R_VAR.L_ELBOW] = camera_angle
            self.next_pos[R_VAR.BODY] = 0
            self.next_pos[R_VAR.R_ELBOW] = 0
            self.next_pos[R_VAR.R_LEG] = 0
            
        else: # right foot suctioned
            
            self.next_pos[R_VAR.L_LEG] = 0
            self.next_pos[R_VAR.L_ELBOW] = 0
            self.next_pos[R_VAR.BODY] = 0
            self.next_pos[R_VAR.R_ELBOW] = camera_angle
                
        # update positon
        self.update()
        
        # face goal
        self.face_point(self.goal)
        self.update()
    
    def get_image(self):
        """
            returns image from unsuctioned RGB image
            as a cv2 image
        """        
      
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            return self.right_image
                
        else: # right foot suctioned
            return self.left_image
        
    def step(self, length, heading):
        """
            rotates & extends the robot to move to a given length and heading. exactly one foot must be suctioned. 
        """           
        print("control: ",length,"|", heading)
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            
            # curr_heading = get_angle(self.left_foot, self.right_foot)
            
            self.next_pos[R_VAR.L_LEG] = self.next_pos[R_VAR.L_LEG] + heading # - curr_heading
            self.next_pos[R_VAR.L_LEG] = (self.next_pos[R_VAR.L_LEG] + 2*np.pi) % (2*np.pi)
            # self.next_pos[R_VAR.L_ELBOW] = 0
            self.next_pos[R_VAR.BODY] = length # - 0.374323 # offset for min robot length
            # self.next_pos[R_VAR.R_ELBOW] = 0
            
            # move to position
            self.update()
            
            # elbows
            self.next_pos[R_VAR.L_ELBOW] = 0
            self.next_pos[R_VAR.R_ELBOW] = 0
            
            self.update()
            
            
            # change footing
            self.set_suction()
            self.update()
            
            
        else: # right foot suctioned
            
            # curr_heading = get_angle(self.right_foot, self.left_foot)
            
            #self.next_pos[R_VAR.L_ELBOW] = 0
            self.next_pos[R_VAR.BODY] = length #  - 0.374323 # offset for min robot length
            #self.next_pos[R_VAR.R_ELBOW] = 0
            
            self.next_pos[R_VAR.R_LEG] = self.next_pos[R_VAR.R_LEG] + heading # - curr_heading
            self.next_pos[R_VAR.R_LEG] = (self.next_pos[R_VAR.R_LEG] + 2*np.pi) % (2*np.pi)
            
            # move to position
            self.update()
            
            # elbows
            self.next_pos[R_VAR.L_ELBOW] = 0
            self.next_pos[R_VAR.R_ELBOW] = 0
            
            self.update()
            
            # change footing
            self.set_suction()
            self.update()
    
    def distance_to_goal(self):
        """
            from base
        """
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            diff = self.goal.flatten()[[0,1]] - self.left_foot.flatten()[[0,1]] 
            
        else: # right foot suctioned
            diff = self.goal.flatten()[[0,1]] - self.right_foot.flatten()[[0,1]] 
            
        return np.sqrt(diff.T @ diff)
        
    def current_pose(self):
        """
            returns the current x,y of the mounted foot with the heading
        """
        
        output = np.zeros((1,3)).flatten()
        
        if self.curr_pos[R_VAR.L_SUCTION] == 1: # left foot suctioned
            output[[0,1]] = self.left_foot[[0,1]]
            output[2] = get_angle(self.left_foot, self.right_foot)
            
        else: # right foot suctioned
            output[[0,1]] = self.right_foot[[0,1]]
            output[2] = get_angle(self.right_foot, self.left_foot)
            
        return output