import torch
from torch import nn
from .linear import Linear
class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, device=None, dtype=None):
        super().__init__()
        '''
        d_model: int  模型隐藏层维度
        d_ff: int  前馈网络中间层维度
        device: torch.device | None = None  参数存放设备
        dtype: torch.dtype | None = None  参数数据类型
        '''
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.SiLU = lambda x: x * torch.sigmoid(x)
    
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)，⊙ 代表逐元素相乘
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        '''


        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))