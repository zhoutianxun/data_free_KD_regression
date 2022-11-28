import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Regressor, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = nn.Tanh()(x)
        x = self.output(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = nn.Tanh()(x)
        x = self.output(x)
        return x
