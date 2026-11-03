import torch
from torch import nn
from einops import einsum

# Problem (embedding):  Implementing the embedding module (1 分)
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__() # 调用父类初始化
        '''
        num_embeddings: int  number of embeddings
        embedding_dim: int  dimension of the embedding
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        
        num_embeddings: int — 嵌入数量
        embedding_dim: int — 嵌入维度
        device / dtype — 设备与数据类型
        '''
        self.w = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        mean, std = 0, 1
        nn.init.trunc_normal_(self.w, mean=mean, std=std, a=-3, b=3)
        


        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
         Apply the linear transformation to the input
        '''
        y = self.w[x]
        # einsum 的 ... 语法自动处理任意 batch 维度，实现广播计算。
        # y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out") 
        return y