import torch
from torch import nn

class BSplineActivation(nn.Module):
    def __init__(self, num_control_points, degree, range_min, range_max):

        super(BSplineActivation, self).__init__()
        self.num_control_points = num_control_points
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max

        if self.num_control_points < self.degree + 1:
            raise ValueError(f"Number of control points ({self.num_control_points}) must be at least degree + 1 ({self.degree + 1})")

        self.control_points = nn.Parameter(torch.linspace(0, 1, num_control_points))  # Shape: (num_control_points,)

        self.register_buffer('knot_vector', self._initialize_knot_vector())

    def _initialize_knot_vector(self):

        num_knots = self.num_control_points + self.degree + 1

        internal_knots = torch.linspace(0, 1, steps=num_knots - 2 * self.degree)

        knot_vector = torch.cat([
            torch.zeros(self.degree),
            internal_knots,
            torch.ones(self.degree)
        ])

        return knot_vector

    def forward(self, x):

        x_scaled = (x - self.range_min) / (self.range_max - self.range_min)
        x_scaled = torch.clamp(x_scaled, 0.0, 1.0)  # Shape: (batch_size, num_features)

        batch_size, num_features = x_scaled.shape
        device = x.device

        N = torch.zeros(batch_size, num_features, self.num_control_points, device=device)

        for i in range(self.num_control_points):
            N[:, :, i] = ((x_scaled >= self.knot_vector[i]) & (x_scaled < self.knot_vector[i + 1])).float()

        N[:, :, -1] += (x_scaled == 1.0).float()

        # Compute higher-degree basis functions using the Cox-de Boor recursion formula
        for d in range(1, self.degree + 1):
            N_new = torch.zeros_like(N)
            for i in range(self.num_control_points - d):
                denom1 = self.knot_vector[i + d] - self.knot_vector[i]
                denom1 = denom1.clamp(min=1e-5)  # Prevent division by zero
                term1 = ((x_scaled - self.knot_vector[i]) / denom1) * N[:, :, i]
                denom2 = self.knot_vector[i + d + 1] - self.knot_vector[i + 1]
                denom2 = denom2.clamp(min=1e-4)  # Prevent division by zero
                term2 = ((self.knot_vector[i + d + 1] - x_scaled) / denom2) * N[:, :, i + 1]

                N_new[:, :, i] = term1 + term2
            N = N_new

        # Compute the spline value by multiplying basis functions with control points
        # control_points: (num_control_points,)
        # N: (batch_size, num_features, num_control_points)
        # Output: (batch_size, num_features)
        activated = torch.matmul(N, self.control_points)

        return activated