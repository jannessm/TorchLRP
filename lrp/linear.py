import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(Linear, self).forward(input)

        p = kwargs.get('pattern')
        if p is not None: return linear[rule](input, self.weight, self.bias, p)
        else: return linear[rule](input, self.weight, self.bias)

    @classmethod
    def from_torch(cls, lin):
        in_feat = lin.weight.shape[1]
        out_feat = lin.weight.shape[0]

        bias = lin.bias is not None
        module = cls(in_features=in_feat, out_features=out_feat, bias=bias)
        module.load_state_dict(lin.state_dict())

        return module
