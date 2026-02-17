import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.activations import BSplineActivation


class KAN(nn.Module):
    def __init__(self, input_channels=3, num_inner_functions=15, num_outer_functions=19,
                 num_control_points=7, degree=2, range_min=-1, range_max=15):

        super(KAN, self).__init__()
        
        self.input_channels = input_channels
        self.num_inner_functions = num_inner_functions
        self.num_outer_functions = num_outer_functions

        self.conv_l1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_l2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_l3 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=3, stride=1, padding=1)

        self.inner_scales = nn.Parameter(torch.empty(num_inner_functions, input_channels))
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
        x = F.relu(self.conv_l1(x))
        x = F.relu(self.conv_l2(x))
        x = F.relu(self.conv_l3(x))

        # flatten
        x = x.view(x.size(0), -1)               # (B, flat_dim)

        # KAN inner
        x1 = F.linear(x, self.inner_scales, self.inner_biases)  # (B, num_inner)
        x1 = self.lay1(x1)

        # KAN outer
        x2 = F.linear(x1, self.outer_scales, self.outer_biases) # (B, num_outer)
        x2 = self.lay2(x2)

        return x2


class PlainKAN(nn.Module):
    def __init__(self, n_classes, input_dim, kan_1, kan_2,
                 spline_cp, spline_deg, range_min, range_max, dropout_p):
        super().__init__()

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_p)

        from src.models.kan_layer import KANLayer
        
        self.kan1 = KANLayer(
            in_features=input_dim, out_features=kan_1,
            num_control_points=spline_cp, degree=spline_deg,
            range_min=range_min, range_max=range_max,
        )
        self.kan2 = KANLayer(
            in_features=kan_1, out_features=kan_2,
            num_control_points=spline_cp, degree=spline_deg,
            range_min=range_min, range_max=range_max,
        )

        self.classifier = nn.Linear(kan_2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.dropout(x)
        x = self.kan2(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
