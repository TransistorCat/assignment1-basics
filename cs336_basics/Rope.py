import torch
import torch.nn as nn
from einops import rearrange

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        '''
        theta: float  RoPE 的 Θ 值
        d_k: int  待编码的向量维度（一般指多头注意力层里 query 和 key 向量的维度）
        max_seq_len: int  输入的最大序列长度
        device: torch.device | None = None  存储 buffer 的设备
        '''
        self.theta = theta
        assert d_k % 2 == 0, "d_k must be even"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 计算 RoPE 的旋转角度
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype)/ d_k)) # (d_k/2,)
        # 计算每个位置的旋转角度
        angles = torch.einsum('i , j -> i j', torch.arange(max_seq_len, device=device, dtype=dtype), inv_freq) # (max_seq_len, d_k/2)
        # 缓存 cos 和 sin 的值，方便后续查表
        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False
        )
        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x: (..., seq_len, d_k) 输入张量，支持任意批量维度
        token_positions: (..., seq_len) 存储每个 token 的位置下标
        return: (..., seq_len, d_k) 旋转位置编码后的张量，与输入形状完全一致
        '''
        # 根据 Token 位置查表
        cos=self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin=self.sin_cached[token_positions]  # (..., seq_len, d_k/2)
        # 把 x 每两个数字分成一组
        x_reshaped = rearrange(x, '... (d_k2 two) -> ... d_k2 two', two=2) # (..., seq_len, d_k) -> (..., seq_len, d_k//2, 2)
        # 取出每组的两个数字
        x1 = x_reshaped[..., 0]  # (..., seq_len, d_k//2)
        x2 = x_reshaped[..., 1]  # (..., seq_len, d_k//2)
        # 对每组的两个数字进行旋转
        x1_rotated = x1 * cos - x2 * sin  # (..., seq_len, d_k//2)
        x2_rotated = x1 * sin + x2 * cos  # (..., seq_len, d_k//2)
        # 把旋转结果重新组合
        x_rotated = torch.stack((x1_rotated, x2_rotated), dim=-1)  # (..., seq_len, d_k//2, 2)
        # 把最后一维的两个数字重新组合成原来的形状
        x_rotated = rearrange(x_rotated, '... d_k2 two -> ... (d_k2 two)', two=2)  # (..., seq_len, d_k//2, 2) -> (..., seq_len, d_k)
        return x_rotated


if __name__ == "__main__":
    # 测试代码
    theta = 10000.0
    d_k = 4
    max_seq_len = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)