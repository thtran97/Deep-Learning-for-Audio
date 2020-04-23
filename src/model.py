import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, ndim, triplet=False):
        super().__init__()
        self.ndim = ndim
        self.triplet = triplet
        self.conv = nn.Sequential(
            nn.Conv2d(1, ndim, (40, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(),            
            nn.Conv2d(ndim, ndim, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU()            
        )
        self.fc = nn.Sequential(
            nn.Linear(2*ndim, ndim),
            nn.Tanh()
        )
        
    def forward(self, x):
#         print("Input size: ", x.size())
        x = self.conv(x)
#         print("Afer conv layer: ", x.size())
        mean, std = x.mean(-1), x.std(-1)
        x = torch.cat([mean, std], dim=1).squeeze(-1)
#         print("Input of fc: ", x.size())
        x = self.fc(x)
#         print("Output of fc: ", x.size())
        
        if self.triplet:
            x = x.view(3, -1, self.ndim)
        else:
            x = x.view(-1, self.ndim)
#         print("Final output size: ", x.size())
        return x 