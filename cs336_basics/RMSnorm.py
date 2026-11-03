import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        '''
        d_model: int  模型隐藏层的特征维度
        eps: float = 1e-5  数值稳定性极小值，防止分母为 0
        device: torch.device | None = None  参数存放设备
        dtype: torch.dtype | None = None  参数数据类型
        '''
        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入形状: [..., d_model] 支持任意批量维度（如 batch, seq_len），最后一维为特征维度
        输出形状: [..., d_model] 与输入形状完全一致
        '''
        in_dtype = x.dtype
        rms_x=x.to(torch.float32)
        x=x.to(torch.float32)
        rms_x=rms_x.pow(2).mean(dim=-1, keepdim=True)
        rms_x=rms_x+self.eps
        rms_x=rms_x.rsqrt()
        x=x*self.gamma*rms_x
        x=x.to(in_dtype)
        return x