from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .functional import lstm

class Object(object):
    pass

class LSTM(torch.nn.LSTM):
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

    def forward(self, input, hidden = None, explain = False, rule = "all", eps = 0.001, bias_factor = 0.0):
        if not explain or rule == 'gradient':
            return super().forward(input, hidden)
        
        orig_input = input
        if self.bidirectional:
            raise NotImplementedError('bidirectional LSTM not supported for LRP')

        input, hx, batch_sizes, sorted_indices, unsorted_indices, is_batched, batch_dim = self._prepare_input(input, hidden)
        
        input_lens = None
        if batch_sizes is not None:
            input, input_lens = pad_packed_sequence(input)
        
        result = self._forward_explain(input, hx, lstm[rule], input_lens, eps, bias_factor)
        
        # if batch_sizes is None:
        #     result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
        #                       self.dropout, self.training, self.bidirectional, self.batch_first)
        # else:
        #     result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
        #                       self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)
    
    def _forward_explain(self, input: Tensor, hx: Tuple[Tensor, Tensor], rule, input_lens = None, eps = 0.001, bias_factor = 0.0):

        seq_dim = 1 if self.batch_first else 0
        b_dim = 0 if self.batch_first else 1
        
        h, c = hx
        h_out = h.clone()
        c_out = c.clone()
        outs = torch.zeros((input.size(seq_dim), input.size(b_dim), h.size(-1)), dtype=input.dtype, device=input.device)

        for i in range(input.size(seq_dim)):
            out = input[tuple([slice(None)] * seq_dim + [i])]
            
            for l in range(self.num_layers):
                if i == 0:
                    _h, _c = h[l].clone(), c[l].clone()
                
                if input_lens is not None:
                    x = x[input_lens > i]
                
                layer = self._get_params_for_layer(l, eps, bias_factor)
                out, _h, _c = rule(out, _h, _c, layer)
                
                if i == input.size(seq_dim) - 1:
                    h_out[l] = _h.clone()
                    c_out[l] = _c.clone()
            outs[i] = out.clone()
        
        return outs, h_out, c_out

    def _get_params_for_layer(self, layer: int, eps: float, bias_factor: float):
        params = Object()
        
        for p in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
            setattr(params, p, self.get_parameter(f'{p}_l{layer}'))
        setattr(params, 'eps', torch.tensor(eps))
        setattr(params, 'bias_factor', torch.tensor(bias_factor))

        return params
    
    def _prepare_input(self, input: Tensor, hx: Tuple[Tensor, Tensor] = None):
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        if isinstance(input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            if batch_sizes is None:  # If not PackedSequence input.
                if is_batched:
                    if (hx[0].dim() != 3 or hx[1].dim() != 3):
                        msg = ("For batched 3-D input, hx and cx should "
                               f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = ("For unbatched 2-D input, hx and cx should "
                               f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        
        return input, hx, batch_sizes, sorted_indices, unsorted_indices, is_batched, batch_dim
