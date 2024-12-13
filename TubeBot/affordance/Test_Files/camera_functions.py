import numpy as np
import matplotlib.pyplot as plt
import math


def rot3d(axis, theta):
    """
        theta -> angle to rotate by, assumed radians
        axis  -> axis to rotate arround, must be x, y, or z
    """
    if axis.upper() == "X":
        rot = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
    elif axis.upper() == "Y":
        rot = np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]])
    elif axis.upper() == "Z":
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    else:
        raise ValueError("rot3d: axis must be X, Y, or Z")

    return rot
    
def rot2d(theta):
    """
        theta -> angle to rotate by, assumed radians
    """

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

    return rot

def precal_points(x_pixels, y_pixels, x_sweep, y_sweep, height, angle, is_rads=False):
    """
        x_pixels -> pixel count in the horizontal direction
        y_pixels -> pixel count int he vertical direction
        x_sweep  -> full angular range in the horizontal direction
        y_sweep  -> full angular range int he vertical direction
        height   -> vertical offset between the cmaer and the ground
        angle    -> angle between the camera plane and the ground
        is_rads  -> indicates whether the provided angles are in radians or degrees,
                        defualt is in degrees
        
        output   <- 3D numpy array,
                        axis 0 = y pixel position,
                        axis 1 = x pixel position,
                        axis 2 = x/y coordinate pair, with x as index 0 and y as index 1
    """
    
    # parameter conversion
    if not is_rads:
        x_sweep = np.deg2rad(x_sweep)
        y_sweep = np.deg2rad(y_sweep)
        angle = np.deg2rad(angle)
        
    # parameter validation
    if not(0 < angle < np.pi):
        raise ValueError("precal_points: camera angle should be between 0 and 180 degrees, maybe check is_rad")
        
    if not(0 < x_sweep < np.pi) or not(0 < y_sweep < np.pi):
        raise ValueError("precal_points: x_sweep and y_sweep angles should be between 0 and 180 degrees, maybe check is_rad")
        
    if x_pixels <= 0 or y_pixels <=0:
        raise ValueError("precal_points: x_pixels and y_pixels should be greater than 0")
        
    # degree change per pixel
    delta_x = x_sweep/x_pixels
    delta_y = y_sweep/y_pixels
    
    # starting angles, assuming sweep values are from outer pixel 
    #  edge to outer pixel edge
    start_x = (x_sweep - delta_x) / 2 
    start_y = -(y_sweep - delta_y) / 2 
    
    print(np.rad2deg(delta_x), "|", np.rad2deg(start_x))
    print(np.rad2deg(delta_y), "|", np.rad2deg(start_y))
    
    # output allocation
    output = np.zeros((y_pixels, x_pixels, 2))
    
    theta_y = start_y
    for j in range(0, y_pixels):
        
        theta_x = start_x
        for i in range(0, x_pixels):
            # create a slop vector for the current angle combination
            slope_vector = rot3d("x", -angle+theta_y) @ rot3d("z", theta_x) @ np.array([[0, 1, 0]]).T 
            
            # find the intesection with z=0
            t = -height / slope_vector[2]
            
            # calculates the x & y for z=0
            point = slope_vector[[0,1]] * t      
            output[j, i, 0] = point[0]
            output[j, i, 1] = point[1]
            
            # increment horizontal angle
            theta_x -= delta_x
            
        # increment vertical angle
        theta_y += delta_y
        
    return np.flipud(output)

def alternative(camera_param, is_rads=False):
    """
        x_pixels -> pixel count in the horizontal direction
        y_pixels -> pixel count int he vertical direction
        x_sweep  -> full angular range in the horizontal direction
        y_sweep  -> full angular range int he vertical direction
        height   -> vertical offset between the cmaer and the ground
        angle    -> angle between the camera plane and the ground
        is_rads  -> indicates whether the provided angles are in radians or degrees,
                        defualt is in degrees
        
        output   <- 3D numpy array,
                        axis 0 = y pixel position,
                        axis 1 = x pixel position,
                        axis 2 = x/y coordinate pair, with x as index 0 and y as index 1
    """
    x_pixels = camera_param[0]
    y_pixels = camera_param[1]
    x_sweep = camera_param[2]
    y_sweep = camera_param[3]
    height = camera_param[4]
    angle = camera_param[5]
    robot_min_length = camera_param[6]
    robot_max_length = camera_param[7]
    
    # parameter conversion
    if not is_rads:
        x_sweep = np.deg2rad(x_sweep)
        y_sweep = np.deg2rad(y_sweep)
        angle = np.deg2rad(angle)
        
    # parameter validation

    if not ((0 < (angle - y_sweep / 2) and (angle + y_sweep / 2) < np.pi)):
        raise ValueError("precal_points: camera angle wrong, maybe check is_rad")
        
    if not(0 < x_sweep < np.pi) or not(0 < y_sweep < np.pi):
        raise ValueError("precal_points: x_sweep and y_sweep angles should be between 0 and 180 degrees, maybe check is_rad")
        
    if x_pixels <= 0 or y_pixels <=0:
        raise ValueError("precal_points: x_pixels and y_pixels should be greater than 0")
        
    # degree change per pixel
    delta_x = x_sweep/x_pixels
    delta_y = y_sweep/y_pixels
    
    # output allocation
    output = np.zeros((y_pixels, x_pixels, 2))





    # for each pixel
    for i in range(0, y_pixels):

        # total angle below ground-parallel
        theta_y_total = angle + (i - y_pixels / 2) * delta_y

        # is this wrong?
        # calculate y easily
        y = height * np.tan(np.pi/2 - theta_y_total)
        
        for j in range(0, x_pixels):
            #THERE ARE ISSUES WITH THIS
            # project onto x,y plane and calculate an unscaled vector in the direction of y,x
            dx = np.sin(theta_y_total) * np.sin((j - x_pixels / 2) * delta_x)
            dy = np.sin(theta_y_total) * np.cos((j - x_pixels / 2) * delta_x)
            

            # use this vector to recalculate a new theta in the x direction
            theta_x = np.arctan2(dx,dy)

            # calculate the x plosition based on theta_x and y
            x = y * np.tan(theta_x)

            d = math.sqrt(x^2 + y^2)
            if d < robot_max_length and d > robot_min_length:
                output[i][j] = [x,y]
            else:
                output = [-1,-1] # flag for unreachable point

    return output

def transform_points(points, delta_x, delta_y, heading, is_rads=False):
    """
        points  -> points to transform, N x M x 2 numpy array 
        delta_x -> distance to translate in the x direction
        delta_y -> distance to translate in the y direction
        heading -> angel to rotate, CCW from po x axis
        is_rads  -> indicates whether the provided angles are in radians or degrees,
                        defualt is in degrees
        
        output   <- 3D numpy array,
                        axis 0 = y pixel position,
                        axis 1 = x pixel position,
                        axis 2 = x/y coordinate pair, with x as index 0 and y as index 1
    """
    
    # parameter conversion
    if not is_rads:
        heading = np.deg2rad(heading)
        
    # get the rotation matrix
    rotation_matrix = rot2d(heading)
    
    # output allocation
    output = np.zeros(np.shape(points))
    
    # applies the rotation matrix to each point,
    #   might be able to do as one matrix operation, 
    #       but I couldn't figure out how to handle the 3D shape
    for j, row in enumerate(points):
        for i, point in enumerate(row):
            output[j, i] = rotation_matrix @ point
            
    # translates the entire matrix in one step
    translation_matrix = np.array([[[delta_x, delta_y]]])
    output = output + translation_matrix
    
    return output

def plot_2d_points(grid):
    """
    Plots 2D points stored in a 3D array.

    Parameters:
        grid (numpy.ndarray): A 3D array of shape (N, M, 2), where each (i, j) cell contains [x, y] coordinates.
    """
    # Ensure the input is a valid 3D array
    if len(grid.shape) != 3 or grid.shape[2] != 2:
        raise ValueError("Input grid must be a 3D array with shape (N, M, 2)")

    # Flatten the grid to extract all points
    points = grid.reshape(-1, 2)
    x_values = points[:, 0]
    y_values = points[:, 1]
    

    # Plot the points
    plt.scatter(x_values, y_values, color='blue', label='Points')
    #plt.scatter(x_corners, y_corners, color='blue', label='Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Points in a Grid')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    camera_param = [320, 240, 0.9948, 0.7734, 0.25, np.pi/2, 0.42228, 0.59642]
    points = alternative(camera_param, is_rads=True)

    print(points)

    print("top")
    print(points[0][0])
    print(points[0][159])
    print(points[0][319])

    print("center")
    print(points[119][0])
    print(points[119][159])
    print(points[119][319])
    
    print("bottom")
    print(points[239][0])
    print(points[239][159])
    print(points[239][319])

    plot_2d_points(points)

    angle = np.pi/2
    print(angle)
    print(np.tan(angle))