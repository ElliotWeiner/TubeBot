from mpl_toolkits import mplot3d
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from camera_functions import *
 
# distance units dont matter
# angles are in degrees, convertiont o radians happens below
height = 5
angle = 45
x_spread = 60
y_spread = 30

# adding a large number of total points slow the 3D plot down
x_pixels = 20
y_pixels = 20

# plots the points in a 3D space with a floor and the camera wireframe
#  flase still plots the camera frame and the floor, just not the points
PLOT_3D = True

   
# find grid    
points = precal_points(x_pixels, y_pixels, x_spread, y_spread, height, angle)


x_spread = np.deg2rad(x_spread)
y_spread = np.deg2rad(y_spread)
angle = np.deg2rad(angle)
 
fig = plt.figure()
 
# syntax for 3-D projection
axis_3d = fig.add_subplot(1,2,1, projection ='3d')

axis_2d = fig.add_subplot(1,2,2)
 
# base edges
resolution = 1000
edge_length = 3*height
edge1 = np.zeros((3, resolution))
edge1[1, :] = np.linspace(0, edge_length, resolution)

edge2 = np.zeros((3, resolution))
edge2[1, :] = np.linspace(0, edge_length, resolution)

edge3 = np.zeros((3, resolution))
edge3[1, :] = np.linspace(0, edge_length, resolution)

edge4 = np.zeros((3, resolution))
edge4[1, :] = np.linspace(0, edge_length, resolution)

# camera edge creation
## looking down
edge1 = rot3d("z", -x_spread/2) @ edge1
edge2 = rot3d("z", x_spread/2) @ edge2
edge3 = rot3d("z", -x_spread/2) @ edge3
edge4 = rot3d("z", x_spread/2) @ edge4

edge1 = rot3d("x", y_spread/2) @ edge1
edge2 = rot3d("x", y_spread/2) @ edge2
edge3 = rot3d("x", -y_spread/2) @ edge3
edge4 = rot3d("x", -y_spread/2) @ edge4

## camera angle
edge1 = rot3d("x", -angle) @ edge1
edge2 = rot3d("x", -angle) @ edge2
edge3 = rot3d("x", -angle) @ edge3
edge4 = rot3d("x", -angle) @ edge4

## camera height
edge1 = edge1 + np.array([[0, 0, height]]).T
edge2 = edge2 + np.array([[0, 0, height]]).T
edge3 = edge3 + np.array([[0, 0, height]]).T
edge4 = edge4 + np.array([[0, 0, height]]).T

## trim to be above xy plane
edge1 = edge1[:, edge1[2,:] >= 0]
edge2 = edge2[:, edge2[2,:] >= 0]
edge3 = edge3[:, edge3[2,:] >= 0]
edge4 = edge4[:, edge4[2,:] >= 0]
 

# plots ground for viewing reference
verts = [[[10, 0, 0],[10, 20, 0],[-10, 20, 0],[-10, 0, 0]]]
axis_3d.add_collection3d(Poly3DCollection(verts, facecolors='cyan', edgecolors='k', zorder=0))

# camera edges
axis_3d.plot3D(*edge1, 'green', zorder=4)
axis_3d.plot3D(*edge2, 'green', zorder=4)
axis_3d.plot3D(*edge3, 'green', zorder=4)
axis_3d.plot3D(*edge4, 'green', zorder=4)


# plots the grid of points
for j, row in enumerate(points):
    for i, point in enumerate(row):
        if i==0 and j==0:
            if PLOT_3D: axis_3d.plot3D(*point, 0, c="g", marker="*", zorder=3)
            axis_2d.plot(*point, "g*")
            
        elif i==x_pixels-1 and j==y_pixels-1:
            if PLOT_3D: axis_3d.plot3D(*point, 0, c="b", marker="*", zorder=3)
            axis_2d.plot(*point, "b*")
            
        else:
            if PLOT_3D: axis_3d.plot3D(*point, 0, c="r", marker="*", zorder=3)
            axis_2d.plot(*point, "r*")

# plots the left hand edge to check linearity
axis_2d.plot([points[0,0][0], points[y_pixels-1, 0][0]], [points[0,0][1], points[y_pixels-1, 0][1]])

# formating
axis_3d.set_xlabel("X")
axis_3d.set_ylabel("Y")
axis_3d.set_zlabel("Z")

axis_3d.set_xlim([-10, 10])
axis_3d.set_ylim([0, 20])
axis_3d.set_zlim([0, 20])

axis_3d.set_box_aspect([1,1,1])

ylim = sum(axis_2d.get_ylim())/2
xlim = sum(np.abs(axis_2d.get_xlim()))/2
axis_2d.set_ylim([ylim-xlim, ylim+xlim])
axis_2d.set_aspect(1)

plt.show()