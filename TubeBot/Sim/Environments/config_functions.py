import warnings
import numpy as np

def subdivide_joint_path(start_config, end_config, n, include_end=True):
    """
        Creates an array of configs to go from the start to the end in n number of steps. The
            configs will be linear in the configuration space, not in the workspace. The function 
            only expects the joint, suction values will end up becoming decimals.
        
        start_config -> starting configuration, intended to be a 1x5 numpy array of the 
                            robot joint values, but will work for any 1xD array
                            
        end_config   -> ending configuration, intended to be a 1x5 numpy array of the 
                            robot joint values, but will work for any 1xD array
        
            n        -> number of steps to create, includes the start and end config
                            
        include_end  -> whether to included the end_config as a part of the output, defualt is True
                            
        
        output       <- the numpy array of configurations, size will be (n+2)xD. D is expected 
                            to be 5, see above.
    """
    
    # validates the value of n
    if n < 1:
        raise ValueError("subdivide_joint_path: n should be at least 1")
    
    # converts the configs to numpy array, if they aren't already
    start_config = np.array(start_config)
    end_config = np.array(end_config)
    
    # validates the input dimensionality
    if max(start_config.shape) != start_config.size:
        raise ValueError("subdivide_joint_path: start config should vector, ie only one dimension should be greater than 1")
        
    if max(end_config.shape) != end_config.size:
        raise ValueError("subdivide_joint_path: end config should vector, ie only one dimension should be greater than 1")
        
    # flattens the configs to 1 dimension
    start_config = start_config.flatten()
    end_config = end_config.flatten()
    
    # validates that the config are the same size
    if start_config.size != end_config.size:
        raise ValueError("subdivide_joint_path: the start and end configurations need to have the same number of elements")
        
    return np.linspace(start_config, end_config, n, endpoint=include_end)
    
    
"""
    The function below assume that the robot joint/suction are in the folowing order:
    
        left_suction, left_leg, left_elbow, body, right_elbow, right_leg, right_suction

"""
    
def camera_pose_path(start_config, mounted_foot, steps=3, angle2ground=45):
    """
        Creates a path to move from the current pose to a posse with the camera in the required 
            position. Assumes the robot has both feet on the ground in curr_config.
            
        The function assumes that the config is structures as:
            left_suction, left_leg, left_elbow, body, right_elbow, right_leg, right_suction
        
        start_config     -> the starting configureation of the robot, made up of 5 joints 
                                poitions and 2 suction toggles
        
        mounted_foot    -> 0 for left foot is currently mounted to the ground, 
                                1 (spcifically non-zero) for right foot is currently 
                                mounted to ground
                                
        angle_to_ground -> the angle between the camera plan and the ground, defaults to 45
                                (included for flexibility)
                                
        steps           -> the minimum numer of steps to use for a joint movement, 
                                ie 1 joint gets steps steps for a movement, 2 joints get 2*steps
                                (included for flexibility)
                                
                                
        output          <- numpy array of cinfigurations, with each row being a config
    """
    
    # flattens the config to 1 dimension
    start_config = np.array(start_config, dtype="float64").flatten()
    
    # validates the start config        
    if start_config.size != 7:
        raise ValueError("camera_pose_path: start config should have 7 values")
    
    # validates steps
    if steps < 1:
        steps = 1
        
    # calculates angles for mounted and un-mounted elbows
    #   numbers are robot lengths in inches
    angle2ground = np.deg2rad(angle2ground)
    unmounted_ang = np.arcsin((11.260589 * (1-np.sin((np.pi/2) - angle2ground))) / 14.737132)
    mounted_ang = angle2ground + unmounted_ang
        
        
    # converts mounted foot into a bool, w/ left as True and 1 and False
    mounted_foot = mounted_foot == 0
        
    output = np.zeros((4*steps,5))
    
    # raise mounted elbow
    curr_config = np.copy(start_config[1:6])
    next_config = np.copy(start_config[1:6])
    next_config[1 if mounted_foot else 3] = mounted_ang
    
    output[0:steps, :] = subdivide_joint_path(curr_config, next_config, steps, include_end=False)
    
    # adjust body length
    curr_config = np.copy(next_config)
    next_config[2] = 0 
    
    output[steps:2*steps, :] = subdivide_joint_path(curr_config, next_config, steps, include_end=False)
    
    # adjust un-mounted elbow and leg
    curr_config = np.copy(next_config)
    next_config[3 if mounted_foot else 1] = unmounted_ang
    next_config[0 if mounted_foot else 4] = 0
    
    output[2*steps:, :] = subdivide_joint_path(curr_config, next_config, 2*steps)

    # activate suction on the mounted side & deactivate on the other side
    if mounted_foot: # mount is on left
        output = np.hstack( ( np.ones((4*steps,1)), output, np.zeros((4*steps, 1)) ) )
        
    else: # mount is on right
        output = np.hstack( ( np.zeros((4*steps, 1)), output, np.ones((4*steps,1)) ) )
    
    # add the original config
    output = np.vstack((start_config, np.hstack((1,start_config[1:6],1)), output))
    
    return output
    
def face_goal_path(start_config, mounted_foot, mounted_position, unmounted_position, goal_position, steps=0):
    """
        Creates a path to orient the robot facing the goal. Assumes the robot is in a camera pose
        
        start_config     -> the starting configureation of the robot, made up of 5 joints 
                                poitions and 2 suction toggles
                
        mounted_foot     -> 0 for left foot is mounted, 1 (spcifically non-zero) for
                                right foot is mounted
                                
        mounted_position -> numpy array for the poition of the mounted foot. z value 
                                can be included, but will be ignored
                                
        unmounted_position -> numpy array for the poition of the un-mounted foot. z value 
                                can be included, but will be ignored
                                
        goal_position    -> numpy array for the poition of the goal. z value can 
                                be included, but will be ignored
        
        steps            -> number of intermeadiate positons to include in the path, 
                                defaults to 0
        
        output           <- a numpy array for a path of configs to aim towards the goal, 
                                size will be (steps+2)x7
    """
    
    # flattens the given coordinates and config
    mounted_position = np.array(mounted_position).flatten()[0:2]
    unmounted_position = np.array(unmounted_position).flatten()[0:2]
    goal_position = np.array(goal_position).flatten()[0:2]
    start_config = np.array(start_config, dtype="float64").flatten()
    
    # validates the start config        
    if start_config.size != 7:
        raise ValueError("face_goal_path: start config should have 7 values")
        
    # validates the number of steps    
    if steps < 0:
        steps == 0
    
    # converts mounted foot into a bool, w/ left as True and 1 and False
    mounted_foot = mounted_foot == 0
    
    # calculates the headings
    point = goal_position - mounted_position
    if point[0] != 0:
        new_heading = abs(np.arctan(point[1]/point[0]))
    else:
        new_heading = np.pi/2
        
    if point[0] < 0:
        new_heading = np.pi - new_heading
        
    if point[1] < 0:
        new_heading = 2*np.pi - new_heading


    point = unmounted_position - mounted_position
    if point[0] != 0:
        curr_heading = abs(np.arctan(point[1]/point[0]))
    else:
        curr_heading = np.pi/2
        
    if point[0] < 0:
        curr_heading = np.pi - curr_heading
        
    if point[1] < 0:
        curr_heading = 2*np.pi - curr_heading
    
    final_config = np.copy(start_config)
    final_config[1 if mounted_foot else 5] = final_config[1 if mounted_foot else 5] + new_heading - curr_heading
    
    # bounds the angle in the [0, 2pi) range and gives a correcting offset
    final_config[1 if mounted_foot else 5] = (final_config[1 if mounted_foot else 5] + (2*np.pi)) % (2*np.pi)  - np.pi/2
    
    return subdivide_joint_path(start_config, final_config, steps+2)
    
def heading_length_path(start_config, mounted_foot, heading, length, steps=6):
    """
        Creates a path to place the un-mounted foot at the relative length and 
            heading. Assumes the robot starts from a camera pose
        
        start_config -> the starting configureation of the robot, made up of 5 joints 
                                poitions and 2 suction toggles
                
        mounted_foot -> 0 for left foot is mounted, 1 (spcifically non-zero) for
                                right foot is mounted
                                
        heading      -> desired relative angle between the two feet, CCW positive from the 
                            positive x-axis of the camera, assumed to be in radians
                                
        length       -> length between the two feet, units are assumed to be compatible with the robot, ie meters
        
        steps        -> number of intermeadiate positons to include in the path, defaults to 6
    """
    
    # flattens the given config
    start_config = np.array(start_config, dtype="float64").flatten()
    
    # validates the start config        
    if start_config.size != 7:
        raise ValueError("face_goal_path: start config should have 7 values")
        
    # validates the length
    if length < 0.374323: # min robot length
        warnings.warn("heading_length_path: requested length is less than miniumn capable of the robot", stacklevel=2)
    elif length > 0.548468: # max robot length
        warnings.warn("heading_length_path: requested length is more than the maximum capable of the robot", stacklevel=2)
        
    # validates the number of steps    
    if steps < 0:
        steps == 6
        
    # converts mounted foot into a bool, w/ left as True and 1 and False
    mounted_foot = mounted_foot == 0
    
    final_config = np.copy(start_config)
    
    # leg joint
    final_config[1 if mounted_foot else 5] = final_config[1 if mounted_foot else 5] - np.pi/2 + heading # might need to add offset val
    
    # elbow joints
    final_config[2] = 0
    final_config[4] = 0
    
    # body joint
    final_config[3] = length - 0.374323 # offset for min robot length
    
    return subdivide_joint_path(start_config, final_config, steps+2)