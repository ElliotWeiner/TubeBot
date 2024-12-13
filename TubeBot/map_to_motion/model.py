import torch
import torch.nn as nn
import numpy as np
import hiera
import timm

# hiera encoder definition
class HieraEncoder(nn.Module):
    def __init__(self, hiera):
        super(HieraEncoder, self).__init__()
        self.hiera = hiera

        # freeze weights
        for weights in self.hiera.parameters():
            weights.requires_grad = False

    def forward(self, x):
        return self.hiera(x)

# full actor definition
class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        # conv 240x320x2  to 128x15x20
        hiera_model = timm.create_model('hiera_base_224', pretrained=True)

        # only get encodings
        hiera_model.head = nn.Identity()
        self.conv = HieraEncoder(hiera_model)

        # predict length & theta
        self.output = nn.Sequential(
            nn.Linear(37632 + 4, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  
        )
        

    def forward(self, affordance, init_point, goal_point):
        # input poits to torch tensors
        init_point = torch.tensor(init_point, dtype=torch.float32).unsqueeze(0)
        goal_point = torch.tensor(goal_point, dtype=torch.float32).unsqueeze(0)
        image = torch.from_numpy(affordance).float()
        image = image.unsqueeze(0).unsqueeze(0)
        # adjust shape for hiera rgb input
        image = image.repeat(1,3,1,1)

        # conv layers
        conv = self.conv(image)

        # flatten
        conv = conv.view(conv.size(0), -1)

        # concat flat conv output with init and goal points
        middle = torch.cat((conv, init_point, goal_point), dim=1)

        # fully connected
        out = self.output(middle)

        return out[0][0], out[0][1]
