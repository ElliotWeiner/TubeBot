import numpy as np
import math
import matplotlib.pyplot as plt

class reward():
    # should my reward be scaled so later steps aren't overly important

    def __init__(self):
        pass
        
    def start(self, start, goal_pos):
        self.previous_distance = math.sqrt((goal_pos[0] - start[0])**2 + (goal_pos[1] - start[1])**2 )

    def value(self, affordance, aff_map, new_pos, goal_pos, base):
        # point affordance value
        p_estimate = self.calc_actual_p(affordance, aff_map, new_pos, base)
        max_affordance = affordance.max()

        print(f"p estimate: {p_estimate}")


        # distance to goal, use to calculate advancement
        distance = math.sqrt((goal_pos[0] - new_pos[0])**2 + (goal_pos[1] - new_pos[1])**2 )

        advancement = self.previous_distance - distance
        max_advancement = 0.548468
        distance_capped = min(self.previous_distance, max_advancement)
        
        self.previous_distance = distance
            
        
        # calculate distance to best for advancement and affordance
        r1 = (distance_capped - advancement) / (distance_capped + max_advancement + 0.0000001)
        r2 = (max_affordance - p_estimate) / (max_affordance + 0.0000001)
        
        # MSE
        r = r1**2+r2**2

        print(f"loss: {r1} + {r2} = {r})")
        print()
        
        return r

    # calculate actual affordance value based on mapping from x,y,z to angleX, angleY
    def calc_actual_p(self, affordance, aff_map, new_pos, base):
        # find x,y closest to new_pos

        # find correct height
        y_best = 10000
        y_idx = -1
        print(f"new_point: {new_pos}")
        
        min_dist = 10000
        best = [0,0]
        for i in range(0, len(aff_map)):
            for j in range(0, len(aff_map[0])):
                pt = aff_map[i][j]
                dist = math.sqrt((pt[0] - new_pos[0])**2 + (pt[1] - new_pos[1])**2 )
                
                if dist < min_dist:
                    best = [i,j]
                    min_dist = dist
        
        
        #self.display(aff_map, aff_map[best[0]][best[1]], new_pos, base)
        return affordance[best[0]][best[1]]
        
    def display(self, grid, point, pt, init):
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
        plt.scatter(point[0], point[1], color='red', label='Point')
        plt.scatter(pt[0], pt[1], color='red', label='Point')
        plt.scatter(init[0], init[1], color='green', label='Point')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Points in a Grid')
        plt.legend()
        plt.grid(True)
        plt.show()


    if __name__ == '__main__':
        camera_param = [320, 240, 0.9948, 0.7734, 0.25, np.pi/4]
        points = precal_points(camera_param, is_rads=True)
        display(points)
