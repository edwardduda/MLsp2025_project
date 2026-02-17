import torch
import torch.nn as nn
from src.models.kan_layer import KANLayer


class KANCNN(nn.Module):
    def __init__(self, n_classes, input_shape, ch1, ch2, ch3, ch4,
                 kan_1, kan_2, spline_cp, spline_deg,
                 range_min, range_max, dropout_p):
        super().__init__()
        C_in, *_ = input_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(C_in, ch1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch3, ch4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        flat_dim = ch4
        self.dropout = nn.Dropout(dropout_p)

        self.kan1 = KANLayer(
            in_features=flat_dim, out_features=kan_1,
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).view(x.size(0), -1)

        x = self.kan1(x)
        x = self.kan2(x)

        logits = self.classifier(x)
        return logits
