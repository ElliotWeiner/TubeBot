# This file contains functions for affordance calculation




import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from affordance.camera_functions import precal_points, transform_points, display
import math


class affordance():

    def __init__(self):
        ## switched from 240 x 320
        self.height = 224
        self.width = 224
        self.edges = 0
        
        self.block = 0
        self.center = 0

        # import map
        self.premap = []

        self.make_map()

    def get_affordance(self, image, c_pose, verbose=False):
        # resize image if necessary (remove later)
        image = self.resize(image)
        _ = self.edge_image(image)
        

        # calculate affordance
        _, aff_img = self.graininess(verbose=verbose)
        
        if verbose:
            cv2.imshow("Stages", aff_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # trim unachieveable points
        self.trim(aff_img)
        
        if verbose:
            cv2.imshow("Stages", aff_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # transform based on c_pose
        #print("c_pose: ",c_pose)
        
        #display(self.premap)
        
        premap = transform_points(self.premap, 0, 0, c_pose[2])
        #print("disp 1")
        #display(premap)

        
        premap = transform_points(premap, c_pose[0], c_pose[1], 0)
        #print("disp 2")
        #display(premap)
        

        # display
        if verbose:
            cv2.imshow("Stages", aff_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return aff_img, premap

    # resize the image
    def resize(self, image):
        new_width = 224
        new_height = 224
        ## switch from 320 x 240

        # resize the image
        return cv2.resize(image, (new_width, new_height))

    # get an absolute value of pixel differences in 4 directions
    def edge_image(self, image):
        #self.height = len(image)
        #self.width = len(image[0])

        # grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)

        # make 2d edge detection copy of image (1 channel)
        self.edges = np.full((self.height, self.width), 255, dtype=np.float32)

        # ignore edges
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                self.edges[i][j] = (
                      abs(gray[i][j] - gray[i][j+1])
                    + abs(gray[i][j] - gray[i][j-1])
                    + abs(gray[i][j] - gray[i+1][j])
                    + abs(gray[i][j] - gray[i-1][j])
                )

        # scale and change dtype
        self.edges = self.edges / 4
        self.edges = self.edges.astype('uint8')

        return self.edges

    # image transformation to prioritize longer and centered steps
    # assumes robot is already pointing towards goal    
    def transformation(self, im):

        image = self.block @ im.copy() @ self.center
        return image
    
    # create blocks for image transformations
    def init_block(self):
        if type(self.block) is int:
            
            self.block = np.zeros((self.height, self.height))
            self.center = np.zeros((self.width, self.width))
            
            vector = np.linspace(1, 0, num=self.height) 
            for i in range(0, self.height):
                self.block[i][i] = vector[i]    
                
            distribution = np.linspace(0.25, 1, num=int(self.width / 2), axis=0) 
            distribution = np.concatenate((distribution, distribution[::-1]))
            for i in range(0, self.width):
                self.center[i][i] = distribution[i]    

    # calculate graininess of surfaces
    def graininess(self, verbose=False):
        
        self.init_block()
        

        # make 2d copy of image (1 channel)
        total_grain = np.zeros((self.height, self.width), dtype=np.float32)

        # set threshold
        x = 100
        
        # for each pixel
        for i in range(0, self.height):
            progress = i / self.height
            if verbose:
                print(f"Progress: {progress:0.3f}", end='\r')
            for j in range(0, self.width):

                #if statement using masked to decide whether to check pixel or not
                
                # for rad multiplying by 5
                for rad in range(1,6): #lower rad to reduce runtime

                    value = 0

                    #ignore edges
                    if i - rad >= 0 and i + rad < self.height:
                        if j - rad >= 0 and j + rad < self.width:

                            #sum edges within square of sidelength 2*rad
                            for pi in range(i - rad, i + 1 + rad):
                                for pj in range(j - rad, j + 1 + rad):

                                    value += self.edges[pi][pj]                

                            if value < x:
                                #best
                                total_grain[i][j] = 50 * rad
                            else: # if limit found break to reduce runtime
                                break
                    
    
        return total_grain.astype('uint8'), self.transformation(total_grain).astype('uint8')

    # make point map of i,j to x,y with preset robot parameters (in function)
    def make_map(self):
        x_sweep = 0.9948
        y_sweep = 0.7734
        height = 0.153475
        angle = 1.222 # 35
        self.robot_min_length = 0.374323
        self.robot_max_length = 0.548468
        offset = 0.448127
        
        camera_param = [self.width, self.height, x_sweep, y_sweep, height, angle]
        self.premap = precal_points(camera_param, is_rads=True)

        self.premap = transform_points(self.premap, offset, 0, -np.pi/2)

    # remove unreachable points
    def trim(self, aff_map):
        #display(self.premap)
        for i in range(0, self.height):
            for j in range(0, self.width):
                # if the distance is too far
                distance = math.sqrt(self.premap[i][j][0]**2 + self.premap[i][j][1]**2)

                if distance > self.robot_max_length:
                    aff_map[i][j] = 0

                elif distance < self.robot_min_length:
                    aff_map[i][j] = 0
