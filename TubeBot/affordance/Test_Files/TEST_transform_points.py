import numpy as np
import math
import matplotlib.pyplot as plt
from camera_functions import *
 

"""
    Green start is array position [0,0] and blue start is [-1, -1]
    Black is original point, red is transformed points
"""


# distance units dont matter
height = 5
angle = 45
x_spread = 60
y_spread = 30

x_pixels = 20
y_pixels = 20

x_slide = 10
y_slide = -10
heading = 135

# find grid    
points = precal_points(x_pixels, y_pixels, x_spread, y_spread, height, angle)

new_points = transform_points(points, x_slide, y_slide, heading)
 
fig, axis = plt.subplots()


# plot original points
for j, row in enumerate(points):
    for i, point in enumerate(row):
        if i==0 and j==0:
            plt.plot(*point, "g*", zorder=3)
            
        elif i==x_pixels-1 and j==y_pixels-1:
            plt.plot(*point, "b*", zorder=3)
            
        else:
            plt.plot(*point, "k*", zorder=3)

# plot new points
for j, row in enumerate(new_points):
    for i, point in enumerate(row):
        if i==0 and j==0:
            plt.plot(*point, "g*", zorder=3)
            
        elif i==x_pixels-1 and j==y_pixels-1:
            plt.plot(*point, "b*", zorder=3)
            
        else:
            plt.plot(*point, "r*", zorder=3)

axis.set_aspect(1)

plt.show()