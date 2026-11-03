import torch
from torch import nn
from einops import einsum

# Problem (linear):  Implementing the linear module (1 分)
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() # 调用父类初始化
        '''
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        
        in_features: int — 输入最后一维的维度
        out_features: int — 输出最后一维的维度
        device / dtype — 设备与数据类型
        '''
        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        mean, std = 0, (2/(in_features+out_features)) ** 0.5
        nn.init.trunc_normal_(self.w, mean=mean, std=std, a=-3*std, b=3*std)
        


        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
         Apply the linear transformation to the input
        '''
        y = x @ self.w.T
        # einsum 的 ... 语法自动处理任意 batch 维度，实现广播计算。
        # y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out") 
        return y