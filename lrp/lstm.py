import torch
from torch import nn
import torch.nn.functional as F
from .functional import lstm

class LSTM(torch.nn.LSTM): 
    def _forward_explain(self, input, hidden, rule_fn, max_len, eos):
        outputs = []
        return rule_fn()


    def forward(self, input, hidden = None, explain=False, rule="all", max_len = None, eos = 0):
        if not explain or rule == 'gradient':
            return super(nn.LSTM, self).forward(input, hidden)
        return self._forward_explain(input, hidden, lstm[rule], max_len, eos)

    @classmethod
    def from_torch(cls, lstm: nn.LSTM):
        args = [
            lstm.input_size,
            lstm.hidden_size,
            lstm.num_layers,
            lstm.bias,
            lstm.batch_first,
            lstm.dropout,
            lstm.bidirectional,
            lstm.proj_size
        ]
        
        module = cls(*args)

        module.load_state_dict(lstm.state_dict())

        return module
