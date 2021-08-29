import sys
sys.path.append('/Users/cy/studyspace/SportsLottery/')

import torch
from torch import nn
from torch.nn import Linear, Transformer
from models.basemodel import BaseModel


import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class Time2Vector(nn.Module):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        self.weights_linear = nn.Linear(seq_len, seq_len, bias=True)
        self.weights_periodic = nn.Linear(seq_len, seq_len, bias=True)

    def forward(self, x):
        x = torch.mean(x, 2)  # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear(x)
        time_linear = torch.unsqueeze(time_linear, 2)  # (batch, seq_len, 1)

        time_periodic = torch.sin(self.weights_periodic(x))
        time_periodic = torch.unsqueeze(time_periodic, 2)  # (batch, seq_len, 1)
        return torch.cat([time_linear, time_periodic], -1)  # (batch, seq_len, 2)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, nums_in_feature: int = 5, seq_len: int = 128):
        super().__init__()
        self.pos_encoder = Time2Vector(seq_len)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(nums_in_feature+2, d_model)
        self.d_model = d_model
        # self.decoder = nn.Linear(d_model, 3)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            # src: Tensor, shape [seq_len, batch_size, nums_in_feature]
            src: Tensor, shape [batch_size, seq_len, nums_in_feature]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = torch.cat([self.pos_encoder(src), src], -1)
        src = self.encoder(src) * math.sqrt(self.d_model)
        transformer_encoder_input = src.permute(1, 0, 2)
        print(transformer_encoder_input.shape)
        output = self.transformer_encoder(transformer_encoder_input, src_mask)
        # output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


if __name__ == '__main__':
    nums_in_feature = 5
    d_model = 256
    nhead = 4
    d_hid = 512
    nlayers = 6
    dropout = 0.1
    seq_len = 128
    batch_size = 32

    model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout)
    input = torch.rand(batch_size, seq_len, nums_in_feature)
    print(model(input, generate_square_subsequent_mask(seq_len)).shape)
