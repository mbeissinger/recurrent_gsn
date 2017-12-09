"""
LSTM with inputs, hiddens, outputs
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
        num_layers=2, bias=True, batch_first=False,
        dropout=0, bidirectional=False, output_activation=nn.Sigmoid()):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, bias=bias)
        self.out_act = output_activation

    def forward(self, x):
        output, _ = self.lstm(x)
        return torch.stack([self.out_act(self.linear(out_t)) for out_t in output])
