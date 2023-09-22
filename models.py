import torch.nn.functional as F
import torch
import torch.nn as nn

class CrossEncoderWithTime(nn.Module):
    def __init__(self, time_dimension=64):
        super(CrossEncoderWithTime,self).__init__()
        triple_embedding_size = 768
        self.fusing = nn.Linear(triple_embedding_size + time_dimension, 64)
        self.linear = nn.Linear(64,1)

    def forward(self, triple_encoding, time):
        quadruple = torch.cat((triple_encoding, time), axis=1)
        x = torch.relu(self.fusing(quadruple))
        x = self.linear(x)
        return x
