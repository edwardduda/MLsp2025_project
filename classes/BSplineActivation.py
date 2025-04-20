import torch
import torch.nn as nn
import torch.nn.functional as F

class BSplineActivation(nn.Module):
    def __init__(self, num_control_points, degree, range_min, range_max):
        """
        B-spline Activation Function.

        Args:
            num_control_points (int): Number of control points for the B-spline.
            degree (int): Degree of the B-spline.
            range_min (float): Minimum value of the input range.
            range_max (float): Maximum value of the input range.
        """
        super(BSplineActivation, self).__init__()
        self.num_control_points = num_control_points
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max

        # Ensure that num_control_points >= degree + 1
        if self.num_control_points < self.degree + 1:
            raise ValueError(f"Number of control points ({self.num_control_points}) must be at least degree + 1 ({self.degree + 1})")

        # Initialize control points (learnable parameters)
        self.control_points = nn.Parameter(torch.linspace(0, 1, num_control_points))  # Shape: (num_control_points,)

        # Initialize knot vector as a buffer (non-trainable)
        self.register_buffer('knot_vector', self._initialize_knot_vector())

    def _initialize_knot_vector(self):
        """
        Initializes a uniform clamped knot vector.

        Returns:
            torch.Tensor: The knot vector.
        """
        # Total number of knots
        num_knots = self.num_control_points + self.degree + 1

        # Uniform internal knots
        internal_knots = torch.linspace(0, 1, steps=num_knots - 2 * self.degree)

        # Clamped knot vector with multiplicity at the ends
        knot_vector = torch.cat([
            torch.zeros(self.degree),
            internal_knots,
            torch.ones(self.degree)
        ])

        return knot_vector  # Shape: (num_knots,)

    def forward(self, x):
        x_scaled = (x - self.range_min) / (self.range_max - self.range_min)
        x_scaled = torch.clamp(x_scaled, 0.0, 1.0)  # (batch_size, num_features)
        batch_size, num_features = x_scaled.shape
        device = x.device

        x_exp = x_scaled.unsqueeze(-1)

        lower = self.knot_vector[:self.num_control_points].view(1, 1, -1)
        upper = self.knot_vector[1:self.num_control_points+1].view(1, 1, -1)
        
        # Indicator: 1 if x in [lower, upper), else 0.
        N = ((x_exp >= lower) & (x_exp < upper)).float()
        
        
        N[:, :, -1] += (x_scaled == 1.0).float()
        
        # ----------- Recursion: Vectorized Cox-de Boor -----------
        # We'll iteratively compute higher-degree basis functions.
        current_num = self.num_control_points  # Current number of basis functions.
        # With each degree, the number reduces by 1.
        for d in range(1, self.degree + 1):
            new_num = current_num - 1  # Number of basis functions for this degree.
            
            # For term1, we need knots u_i and u_{i+d} for i=0,...,new_num-1.
            knot_i     = self.knot_vector[:new_num].view(1, 1, -1)            # (1,1,new_num)
            knot_i_d   = self.knot_vector[d: d + new_num].view(1, 1, -1)        # (1,1,new_num)
            denom1     = (knot_i_d - knot_i).clamp(min=1e-5)                    # (1,1,new_num)
            
            # For term2, we need knots u_{i+1} and u_{i+d+1} for i=0,...,new_num-1.
            knot_i1    = self.knot_vector[1:new_num+1].view(1, 1, -1)            # (1,1,new_num)
            knot_i_d1  = self.knot_vector[d+1: d + new_num + 1].view(1, 1, -1)    # (1,1,new_num)
            denom2     = (knot_i_d1 - knot_i1).clamp(min=1e-5)                   # (1,1,new_num)
            
            # Compute the two terms using broadcasting over batch and features.
            term1 = ((x_exp - knot_i) / denom1) * N[..., :new_num]   # (batch_size, num_features, new_num)
            term2 = ((knot_i_d1 - x_exp) / denom2) * N[..., 1:new_num+1]  # (batch_size, num_features, new_num)
            
            # Sum the terms to get the new basis functions.
            N = term1 + term2  # (batch_size, num_features, new_num)
            current_num = new_num
        
        # ----------- Final Activation -----------
        # After recursion, N has shape (batch_size, num_features, current_num)
        # current_num should equal (num_control_points - degree), which is the effective number of basis functions.
        # We assume here that our control points align with these basis functions.
        ctrl = self.control_points[:current_num].view(1, 1, -1)  # (1,1,current_num)
        # Multiply basis functions with control points and sum along the last dimension.
        activated = torch.sum(N * ctrl, dim=-1)  # (batch_size, num_features)
        
        return activated
