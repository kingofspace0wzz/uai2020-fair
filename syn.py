import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.binomial import Binomial
from torch.distributions.categorical import Categorical

S = Categorical(torch.tensor([0.5, 0.5]))
for i in range(100000):
    s = S.sample()
    x = []
    if s==1:
        u
    else
    