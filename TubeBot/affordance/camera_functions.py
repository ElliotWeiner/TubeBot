import numpy as np
import matplotlib.pyplot as plt

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

def precal_points(camera_param, is_rads=False):
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
        
    output = np.zeros((y_pixels, x_pixels, 2))

    # find the maximum bounds, if the camera is pointing down, using symmetry about z axis
    y_edge = height * np.tan(y_sweep/2)
    x_edge = height * np.tan(x_sweep/2)

    # rotation matrix to adjust for the camera angle
    R = rot3d("x", (np.pi/2)-angle)

    for j, y_val in enumerate(np.linspace(-y_edge, y_edge, y_pixels)):
        for i, x_val in enumerate(np.linspace(-x_edge, x_edge, x_pixels)):
            # creates point with camera at origin, facing down
            point = np.array([x_val,  y_val, -height])

            # rotates the camera
            point = (R @ point)
            
            # shortens the vector so that z=0
            output[j, i, :] = point[[0,1]] * (-height/point[2])
        
    return np.flipud(output)

def transform_points(points, delta_x, delta_y, heading, is_rads=True):
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
        
    # heading -= np.pi/2
        
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


def display(grid):
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
    ## updated from 320, 240 (double check)
    camera_param = [224, 224, 0.9948, 0.7734, 0.25, np.pi/4]
    points = precal_points(camera_param, is_rads=True)
    display(points)