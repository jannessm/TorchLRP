'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

import numpy as np
from numpy import newaxis as na

def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
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
    sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)

    numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
    
    denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)
    
    message  = (numer/denom) * Rout[na,:]       # shape (D, M)
    Rin      = message.sum(axis=-1)              # shape (D,)
    # print(Rin.shape, message.shape)
    
    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note: 
    # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
    # -> can be used for sanity check
    
    return Rin

class LSTM_bidi:
    
    def __init__(self, model):
        """
        Load trained model from file.
        """

        # LSTM left encoder
        self.Wxh  = model.weight_ih_l0.detach().numpy()  # shape 4d*e
        self.bxh  = model.bias_ih_l0.detach().numpy()  # shape 4d 
        self.Whh  = model.weight_hh_l0.detach().numpy()  # shape 4d*d
        self.bhh  = model.bias_hh_l0.detach().numpy()  # shape 4d
    

    def set_input(self, x, delete_pos=None):
        """
        Build the numerical input sequence x/x_rev from the word indices w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """
        T      = x.shape[0]                         # sequence length
        d      = int(self.Wxh.shape[0]/4)  # hidden layer dimension
        e      = 256                # word embedding dimension

        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))
        
        self.w              = list(range(T))
        self.x              = x
        
        self.h         = np.zeros((T+1, d))
        self.c         = np.zeros((T+1, d))
     
   
    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T      = len(self.w)                         
        d      = int(self.Wxh.shape[0]/4) 
        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):     
        idx    = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
          
        # initialize
        self.gates_xh  = np.zeros((T, 4*d))  
        self.gates_hh  = np.zeros((T, 4*d)) 
        self.gates_pre = np.zeros((T, 4*d))  # gates pre-activation
        self.gates     = np.zeros((T, 4*d))  # gates activation
             
        for t in range(T): 
            self.gates_xh[t]     = np.dot(self.Wxh, self.x[t])       
            self.gates_hh[t]     = np.dot(self.Whh, self.h[t-1])
            self.gates_pre[t]    = self.gates_xh[t] + self.gates_hh[t] + self.bxh + self.bhh
            self.gates[t,idx]    = 1.0/(1.0 + np.exp(- self.gates_pre[t,idx]))
            self.gates[t,idx_g]  = np.tanh(self.gates_pre[t,idx_g]) 

            self.c[t]            = self.gates[t,idx_f]*self.c[t-1] + self.gates[t,idx_i]*self.gates[t,idx_g]
            self.h[t]            = self.gates[t,idx_o]*np.tanh(self.c[t])

        self.s       = self.h[T-1]

        return self.s.copy() # prediction scores
    
                   
    def lrp(self, x, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(x)
        self.forward() 
        
        T      = len(self.w)
        d      = int(self.Wxh.shape[0]/4)
        idx_i, idx_f, idx_g = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d) # indices of gates i,g,f separately
        
        # initialize
        Rx  = np.zeros(self.x.shape)
        
        Rh  = np.zeros((T+1, d))
        Rc  = np.zeros((T+1, d))
        Rg  = np.zeros((T,   d)) # gate g only
        
        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh[T-1]  = np.ones((d))
        for t in reversed(range(T)):
            Rc[t]   += Rh[t]

            Rg[t]    = lrp_linear(self.gates[t,idx_i]*self.gates[t,idx_g], np.identity(d), np.zeros((d)), self.c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rx[t]    = lrp_linear(self.x[t], self.Wxh[idx_g].T, self.bxh[idx_g]+self.bhh[idx_g], self.gates_pre[t,idx_g], Rg[t], d+256, eps, bias_factor, debug=False)
            Rc[t-1]  = lrp_linear(self.gates[t,idx_f]*self.c[t-1],         np.identity(d), np.zeros((d)), self.c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rh[t-1]  = lrp_linear(self.h[t-1], self.Whh[idx_g].T, self.bxh[idx_g]+self.bhh[idx_g], self.gates_pre[t,idx_g], Rg[t], d+256, eps, bias_factor, debug=False)
                   
        return Rx
