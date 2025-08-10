import torch.nn as nn
import torch

class SimAM(nn.Module):
    def __init__(self, channels, lambda_=1e-4):
        super(SimAM, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        N, C, H, W = x.shape
        n = H * W - 1

        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean) ** 2
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4 * (v + self.lambda_)) + 0.5
        return x * torch.sigmoid(e_inv)
    
