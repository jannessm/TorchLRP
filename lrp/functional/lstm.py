from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Function
from torch import _VF

class LstmAll(Function):
    
    @staticmethod
    def lstm_cell(x: Tensor,                    # batches x input_size
                 hx: Tuple[Tensor, Tensor],     # (d, d)
                 weight_ih: Tensor,             # 4*d x input_size
                 weight_hh: Tensor,             # 4*d x d
                 bias_ih: Tensor,               # 4*d
                 bias_hh: Tensor                # 4*d
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        prev_h, prev_c = hx
        b = x.size(0)
        d = bias_hh.size(-1) // 4

        gates = torch.zeros((b, bias_hh.size(-1)), dtype=x.dtype, device=x.device)

        idx_i = torch.arange(0, d)
        idx_f = torch.arange(d, 2*d)
        idx_g = torch.arange(2*d, 3*d)
        idx_o = torch.arange(3*d, 4*d)
        idx_ifo = torch.concat([idx_i, idx_f, idx_o])

        gates_xh = weight_ih[None].type(x.dtype).repeat((b,1,1)).matmul(     x[:, :, None]).squeeze(-1) # b x 4d
        gates_hh = weight_hh[None].type(x.dtype).repeat((b,1,1)).matmul(prev_h[:, :, None]).squeeze(-1) # b x 4d
        
        gates_pre = gates_xh + gates_hh + bias_ih + bias_hh                 # b x 4d
        gates[:, idx_ifo] = torch.sigmoid(gates_pre[:, idx_ifo])
        gates[:, idx_g] = torch.tanh(gates_pre[:, idx_g])
        
        c = gates[:, idx_f] * prev_c + gates[:, idx_i] * gates[:, idx_g]    # b x d
        h = gates[:, idx_o] * torch.tanh(c)                                 # b x d

        return gates_pre, gates, h, c

    @staticmethod
    def lrp_linear(hin: Tensor, w: Tensor, b: Tensor, hout: Tensor, Rout: Tensor, bias_nb_units: Tensor, eps: float, bias_factor: float = 0.0, debug=False) -> Tensor:
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - hin:            forward pass input, of shape (D,)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
        - Rout:           relevance at layer output, of shape (M,)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
        Returns:
        - Rin:            relevance at layer input, of shape (D,)
        """
        sign_out = torch.where(hout>=0, 1., -1.) # shape (1, M)

        numer    = (w * hin.T) + ( bias_factor * (b[None,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
        # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
        # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
        # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
        
        denom    = hout + (eps*sign_out*1.)         # shape (1, M)

        message  = (numer/denom) * Rout             # shape (D, M)
        
        Rin      = message.sum(dim=-1)              # shape (D,)

        if debug:
            print("local diff: ", Rout.sum() - Rin.sum())
        # Note: 
        # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
        # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
        # -> can be used for sanity check
        
        return Rin
    
    @staticmethod
    def forward(ctx, inputs: Tensor, h: Tensor, c: Tensor, layer: nn.LSTM):
        assert inputs.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {inputs.dim()}-D tensor"
        is_batched = inputs.dim() == 2
        if not is_batched:
            inputs = inputs.unsqueeze(0)

        hx = (h.unsqueeze(0), c.unsqueeze(0)) if not is_batched else (h, c)

        gates_pre, gates, h, c = LstmAll.lstm_cell(
            inputs, hx,
            layer.weight_ih, layer.weight_hh,
            layer.bias_ih, layer.bias_hh,
        )
        
        ctx.save_for_backward(inputs, hx[0], hx[1], h, c, layer.weight_ih, layer.weight_hh, layer.bias_ih, layer.bias_hh, gates_pre, gates, layer.eps, layer.bias_factor)

        if not is_batched:
            h = (h[0].squeeze(0), h[1].squeeze(0))
        return h.clone(), h.clone(), c


    @staticmethod
    def backward(ctx, R_out: Tensor, R_h: Tensor, R_c: Tensor):
        if (R_h == 0).all():
            R_h = R_out

        device = R_out.device
        
        inputs, h_in, c_in, h_out, c_out, weight_ih, weight_hh, bias_ih, bias_hh, gates_pre, gates, eps, bias_factor = ctx.saved_tensors

        d = bias_hh.size(-1) // 4               # hidden size
        e = weight_ih.size(-1)
        eye = torch.eye(d, device=device, dtype=R_out.dtype)
        zeros = torch.zeros((d,), device=device, dtype=R_out.dtype)

        idx_i = torch.arange(0, d)
        idx_f = torch.arange(d, 2*d)
        idx_g = torch.arange(2*d, 3*d)
        
        bias = bias_ih[idx_g] + bias_hh[idx_g]
        
        # timestep t
        R_c_out = R_c + R_h
        
        gates_ig = gates[:, idx_i] * gates[:, idx_g]
        R_g = LstmAll.lrp_linear(gates_ig,
                                 eye.clone(), zeros.clone(),
                                 c_out,
                                 R_c_out,
                                 d, eps, bias_factor)[None]

        R_x = LstmAll.lrp_linear(inputs,
                                 weight_ih[idx_g].T,
                                 bias,
                                 gates_pre[:, idx_g],
                                 R_g,
                                 d+e, eps, bias_factor, debug=False)[None]

        # timestep t-1
        R_c_in = LstmAll.lrp_linear(gates[:, idx_f] * c_in,
                                 eye.clone(), zeros.clone(),
                                 c_out,
                                 R_c_out,
                                 d, eps, bias_factor)[None]

        R_h_in = LstmAll.lrp_linear(h_in,
                                    weight_hh[idx_g].T,
                                    bias,
                                    gates_pre[:, idx_g],
                                    R_g,
                                    d+e, eps, bias_factor)[None]

        return R_x, R_h_in, R_c_in, None
        

lstm = {
    "gradient": _VF.lstm_cell,
    "all": LstmAll.apply,
}

