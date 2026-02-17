import torch
import torch.nn as nn


class KANLayer(nn.Module):
    """
    A true Kolmogorov-Arnold Network layer.

    Each edge (i -> j) has its own independent learnable B-spline function.
    output_j = sum_i  BSpline(x_i ; control_points[i, j, :])

    There are no weight matrices -- the per-edge spline functions ARE the
    learnable parameters that replace traditional weights.
    """

    def __init__(self, in_features, out_features,
                 num_control_points, degree, range_min, range_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_control_points = num_control_points
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max

        if num_control_points < degree + 1:
            raise ValueError(
                f"num_control_points ({num_control_points}) must be "
                f">= degree + 1 ({degree + 1})"
            )

        num_basis = num_control_points - degree
        self.control_points = nn.Parameter(
            torch.randn(in_features, out_features, num_basis) * 0.1
        )

        self.register_buffer('knot_vector', self._build_knot_vector())

    def _build_knot_vector(self):
        num_knots = self.num_control_points + self.degree + 1
        internal = torch.linspace(0, 1, steps=num_knots - 2 * self.degree)
        return torch.cat([
            torch.zeros(self.degree),
            internal,
            torch.ones(self.degree),
        ])

    def _compute_basis(self, x_scaled):
        """
        Vectorised Cox-de Boor recursion.

        Args:
            x_scaled: (batch, in_features) values in [0, 1]

        Returns:
            N: (batch, in_features, num_basis) basis function values
        """
        eps = torch.finfo(x_scaled.dtype).eps * 100
        x_exp = x_scaled.unsqueeze(-1)

        lower = self.knot_vector[:self.num_control_points].view(1, 1, -1)
        upper = self.knot_vector[1:self.num_control_points + 1].view(1, 1, -1)

        N = ((x_exp >= lower) & (x_exp < upper)).float()

        boundary = (x_scaled == 1.0).float().unsqueeze(-1)
        N[:, :, -1:] += boundary

        current_num = self.num_control_points
        for d in range(1, self.degree + 1):
            new_num = current_num - 1

            ki = self.knot_vector[:new_num].view(1, 1, -1)
            kid = self.knot_vector[d:d + new_num].view(1, 1, -1)
            d1 = (kid - ki).clamp(min=eps)

            ki1 = self.knot_vector[1:new_num + 1].view(1, 1, -1)
            kid1 = self.knot_vector[d + 1:d + new_num + 1].view(1, 1, -1)
            d2 = (kid1 - ki1).clamp(min=eps)

            t1 = ((x_exp - ki) / d1) * N[..., :new_num]
            t2 = ((kid1 - x_exp) / d2) * N[..., 1:new_num + 1]

            N = t1 + t2
            current_num = new_num

        return N

    def forward(self, x):
        """
        Args:
            x: (batch, in_features)

        Returns:
            (batch, out_features)
        """
        x_scaled = (x - self.range_min) / (self.range_max - self.range_min)
        x_scaled = torch.clamp(x_scaled, 0.0, 1.0)

        N = self._compute_basis(x_scaled=x_scaled)

        # N:              (batch, in_features, num_basis)
        # control_points: (in_features, out_features, num_basis)
        # output_j = sum_i sum_k  N[b,i,k] * cp[i,j,k]
        return torch.einsum('bik,iok->bo', N, self.control_points)
