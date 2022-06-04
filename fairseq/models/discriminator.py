import torch
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):
        return (-grad_output)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = GradReverse.apply(x)
        x = self.discriminator(x)
        return x.view(-1)