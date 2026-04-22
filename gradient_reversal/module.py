from .functional import revgrad, scalegrad
import torch
from torch import nn

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32, requires_grad=False))

    def forward(self, x):
        return revgrad(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha.fill_(float(alpha))

# class GradientScaler(nn.Module):
#     def __init__(self, alpha):
#         super().__init__()
#         self.alpha = torch.tensor(alpha, requires_grad=False)

#     def forward(self, x):
#         return (x, self.alpha)

## Not really neccesary ... 
#  Just use:  
##   y = x * alpha + x.detach() * (1 - alpha)
#
