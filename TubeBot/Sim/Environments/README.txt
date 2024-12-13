Static_Enviornment contains the robot in an environment that has 4 types of
 floor and a random scattering of texteured patches

The simulation will automatically launch the SimControl code/node 
 and it will clean the node up when the simulation stops. The SimCOntrol file 
 needs to be int he same directory that the simulation file is in.



SimControl is a wrapper node for controllign the environemnt, and it shouldn't need 
  to be modified. It will takes data from a series of topics and pass them in the 
  neccesary format for the sim.

exampleControl is an example of how to send values to the SimControl node.

SimOutput will create a window that displays the right/left camera images and positions.

Topic List:
	<topic name>  <topic type> < discription>

  Data into the Sim:
	right_leg	Float32		angle of the right leg, in radians
	right_elbow	Float32		angle of the right elbow, in radians
	body		Float32		length of the body extention, in m (0 is fully collapsed, >0 is extended)
	left_elbow	Float32		angle of the right elbow, in radians
	left_leg	Float32		angle of the left leg, in radians
	
	right_suction	Bool		indicates if the right foot is suctioning
	left_suction	Bool		indicates if the left foot is suctioning

	camera_pos	UInt8		indicates which camera pos to move to NOT WORKING YET
						(0 for none, 1 for rightfoot down, 2 for left foot down)

  Data out of the Sim:
	right_Depth_image	Image	right depth camera
	right_RGB_image		Image	right RGB camera
	left_Depth_image	Image	left depth camera
	left_RGB_image		Image	left rgb camera
	
	right_ref		Point	center of the right camera/bottom of the foot
	left_ref		Point	center of the left camera/bottom of the foot

