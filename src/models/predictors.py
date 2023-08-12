import torch
from typing import Optional

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: int = 1, include_edge: bool = False, edge_dim: Optional[int] = 0):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin_src = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_dst = torch.nn.Linear(in_channels, hidden_channels)
        self.include_edge = include_edge
        self.lin_final = torch.nn.Linear(hidden_channels + edge_dim if self.include_edge else hidden_channels, out_channels)

    def forward(self, z_src, z_dst, msg):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        if self.include_edge:
            h = torch.cat((h, msg), dim=-1)
        h = h.relu()
        return self.lin_final(h)