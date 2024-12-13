import torch
from torch.optim import Adam


from map_to_motion.model import Actor  
from map_to_motion.reward import reward 

from affordance.affordance import affordance  

from Sim.sim_functions import sim 
from affordance.camera_functions import precal_points, transform_points, display

import numpy as np
import time
import random

import matplotlib.pyplot as plt

# training loop
def train():

    # init affordance module
    aff_module = affordance()

    # init simulator
    simulator = sim()

    # init policy
    policy = Actor()

    # init reward and optimizer
    loss_function = reward()
    optimizer = Adam(policy.parameters(), lr=1e-4)

    
    # init thresholds
    threshold = 0.5 # goal radius

    max_steps = 5
    max_length = 0.174 # max body length
    epsilon = 1

    losses = []

    # runtime per training series
    runtime = 129600 # 1.5 days
    start = time.time()

    while(True):
        # storage
        rewards = []
        steps = 0

        # environment reset
        goal_pos = simulator.new_goal()
        
        # get current base position
        current_pos = simulator.get_base()
        print(f"Initial pose: {current_pos}")
        
        loss_function.start(current_pos, goal_pos)

        while(True):
           

            if steps < max_steps:

                # set init pose
                simulator.init_pose(camera_angle=20)

                # get picture
                image = simulator.get_image() 
                
                # x y heading current
                c_pose = simulator.current_pose()

                # get affordance
                affordance_map, coord_map = affordance_step(aff_module, image, c_pose)



                # MODEL TRAINING

                # get control from policy
                
                # exploitation
                length, theta = policy.forward(affordance_map, current_pos, goal_pos)
            
                # clip results
                length_new = (length % max_length)
                print(f"clippled {length} to {length_new}")
                length = length_new
                theta = (theta % (2 * np.pi))

                
                if random.random() < epsilon:
                    print('random move')
                    # exploration
                    length = length - length + random.uniform(0, max_length)
                    theta = theta - theta + random.uniform(0, 2* np.pi)
                    
                    while(True):
                        theta = theta - theta + random.uniform(0, 2* np.pi)

                        # bound it to 11 degrees either way, or 0.19 radians (ish)
                        # these are the only steps it should ever be taking, only examples it needs
                        if theta < 0.19:
                            break
                        elif theta > 6.09:
                            break
                    

                # execute control
                
                print(f"extension: {100 * length.item() / max_length}%")
                print(f"rotation: {100 * theta.item() / (2*np.pi)}%")
                
                simulator.step(length.item(), theta.item())


                
                
                # get reward
                r = loss_function.value(affordance_map, coord_map, simulator.get_base(), goal_pos, current_pos)

                # switch current base position
                current_pos = simulator.get_base()
                

                # since reward is explicitly calculated from the sim, must use a proxy reward to distribute loss
                proxy = (length / (length + theta + 0.000001)) + (theta / (length + theta + 0.000001))
                rewards.append(r * proxy)

                steps += 1


                # exit when done
                distance = simulator.distance_to_goal()
                if distance < threshold:
                    # goal reward
                    rewards[-1] = rewards[-1] / 2
                    break

            else:
                break

        # reward function is more of a loss function
        policy_loss = torch.mean(torch.stack(rewards), dim=0)

        print("mean loss: ", policy_loss)
        print('\n')

        losses.append(policy_loss)

        # calculate gradients
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

        optimizer.step()

        epsilon = max(0.1, epsilon * 0.95)

        # after runtime, save model and exit
        if time.time() - start > runtime:
            torch.save(policy.state_dict(), 'model_hiera_36h.pth')
            np.savetxt("losses.csv", losses, delimiter=",", fmt="%f")
            display_loss(losses)
            break

    return True
            

# calculate affordance
def affordance_step(module, image, c_pose):
	aff, coord_map = module.get_affordance(image, c_pose, verbose=False)

	return aff, coord_map

# basic display function
def display_loss(losses):
    plt.plot(losses)
    
    # add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('TubeBot Training')
    
    # add grid
    plt.grid(True)
    
    # display  plot
    plt.show()


if __name__ == '__main__':
    train()
