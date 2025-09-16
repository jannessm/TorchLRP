import torch

class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False, rule="epsilon", pattern=None):
        if not explain:
            return super().forward(input)

        for module in self:
            if 'lrp' in str(module.__class__):
                input = module(input, explain=explain)
            else:
                input = module(input)
        return input

    @classmethod
    def from_torch(cls, seq):
        from .converter import convert
        
        module = cls(*seq)
        convert(module)

        return module
