from virtual_objects import Virtual_Robot
import numpy as np

robot = Virtual_Robot()

robot.set_suction(right=True)
print(robot, "\n")
robot.update()

robot.camera_pose()
print(robot, "\n")
robot.update()

robot.wait(6)

robot.face_point([-5,-5])
print(robot, "\n")
robot.update()

robot.wait(6)

robot.face_point([0,-5])
print(robot, "\n")
robot.update()

robot.wait(6)

robot.heading_length(0.5, np.pi/2)
print(robot, "\n")
robot.update()

print(robot, "\n")
robot.shutdown()

print("Code Done")