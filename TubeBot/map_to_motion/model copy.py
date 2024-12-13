import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):


    def __init__(self):
        super(Actor, self).__init__()

        # conv 240x320x2  to 128x15x20
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # predict length & theta
        self.output = nn.Sequential(
            nn.Linear(256 * 15 * 20 + 4, 512), 
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  
        )
        

    def forward(self, affordance, init_point, goal_point):
        # input poits to torch tensors
        init_point = torch.tensor(init_point, dtype=torch.float32).unsqueeze(0)
        goal_point = torch.tensor(goal_point, dtype=torch.float32).unsqueeze(0)
        image = torch.from_numpy(affordance).float()
        image = image.unsqueeze(0).unsqueeze(0)

        # conv layers
        conv = self.conv(image)

        # flatten
        conv = conv.view(conv.size(0), -1)

        # concat flat conv output with init and goal points
        middle = torch.cat((conv, init_point, goal_point), dim=1)

        # fully connected
        out = self.output(middle)

        return out[0][0], out[0][1]
