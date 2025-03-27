import torch
from torch import nn, F
from BSplineActivation import BSplineActivation
from Config import Config

class KAN(nn.Module):
    def __init__(self, input_dim=784, num_inner_functions=15, num_outer_functions=19,
                 num_control_points=7, degree=2, range_min=-3.401, range_max=33.232):

        super(KAN, self).__init__()
        
        
        self.flatten = nn.Flatten()
        self.input_dim = input_dim
        self.num_inner_functions = num_inner_functions
        self.num_outer_functions = num_outer_functions


        self.inner_scales = nn.Parameter(torch.empty(num_inner_functions, input_dim))
        self.inner_biases = nn.Parameter(torch.empty(num_inner_functions))

        self.outer_scales = nn.Parameter(torch.empty(num_outer_functions, num_inner_functions))
        self.outer_biases = nn.Parameter(torch.empty(num_outer_functions))

        self.lay1 = BSplineActivation(
            num_control_points=num_control_points,
            degree=degree,
            range_min=range_min,
            range_max=range_max
        )
        self.lay2 = BSplineActivation(
            num_control_points=num_control_points,
            degree=degree,
            range_min=range_min,
            range_max=range_max
        )

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        with torch.no_grad():
            nn.init.normal_(self.inner_scales, mean=0.0, std=0.05)
            nn.init.normal_(self.outer_scales, mean=0.0, std=0.05)
            nn.init.zeros_(self.inner_biases)
            nn.init.zeros_(self.outer_biases)

    def forward(self, x):

        x = self.flatten(x)

        scaled_input_lay1 = F.linear(x, self.inner_scales, self.inner_biases)
        lay1_outputs = self.lay1(scaled_input_lay1)

        scaled_input_lay2 = F.linear(lay1_outputs, self.outer_scales, self.outer_biases)
        outer_outputs = self.lay2(scaled_input_lay2) 

        return outer_outputs