
from einops import rearrange
import torch
import torch.nn as nn
from .linear import Linear
from .Rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    '''
    x: 输入张量，支持任意维度
    i: int  待计算 softmax 的维度编号
    返回形状: 与输入 x 形状完全一致
    '''
    # 取出待计算 softmax 的维度的最大值，避免指数运算溢出
    x_max = torch.max(x, dim=i, keepdim=True).values
    # 计算指数
    exp_x = torch.exp(x - x_max)
    # 沿着指定维度求和
    sum_exp_x = torch.sum(exp_x, dim=i, keepdim=True)
    # 相除
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    实现缩放点积注意力
    q, k: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    mask: 可选布尔张量 (seq_len, seq_len)
          mask=True 代表该位置允许参与注意力权重计算, False 权重置零
    return: (batch_size, ..., seq_len, d_v)
    """
    # 计算 Q 和 K 的点积分数
    scores = q @ k.transpose(-2, -1)
    
    # 缩放
    d_k = q.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))


    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    # 计算注意力权重
    attn_weights = softmax(scores, i=-1)

    # 应用注意力权重到 V
    out = attn_weights @ v

    return out




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, rope_theta: float = None) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  注意力头数
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率
        '''
        # 计算每个头的维度
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        assert self.d_k % 2 == 0, "d_k must be even for RoPE"
        self.num_heads = num_heads
        # 定义线性层
        self.qkv_proj = Linear(d_model, 3 * d_model) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, 3 * d_model)
        self.out_proj = Linear(d_model, d_model)
        if rope_theta is not None and max_seq_len is None:
            assert False, "If rope_theta is provided, max_seq_len must also be provided"
        if rope_theta is None and max_seq_len is not None:
            assert False, "If max_seq_len is provided, rope_theta must also be provided"
        if rope_theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, max_seq_len)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        '''
        x: (batch_size, seq_len, d_model)  输入张量
        token_positions: (batch_size, seq_len) 或 None  各 token 的位置下标
        返回形状: (batch_size, seq_len, d_model)
        '''

        q_proj, k_proj, v_proj = torch.chunk(self.qkv_proj(x), 3, dim=-1) 
        # 根据 self.num_heads 将 q_proj, k_proj, v_proj 重新排列为 (batch_size, num_heads, seq_len, d_k)
        range_q = rearrange(q_proj, 'b s (h d) -> b h s d', h=self.num_heads)
        range_k = rearrange(k_proj, 'b s (h d) -> b h s d', h=self.num_heads)
        range_v = rearrange(v_proj, 'b s (h d) -> b h s d', h=self.num_heads)

        # 计算 RoPE
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)  # (batch_size, seq_len)
        
        if hasattr(self, 'rope'):
            rope_q = self.rope(range_q, token_positions)
            rope_k = self.rope(range_k, token_positions)
        else:
            rope_q = range_q
            rope_k = range_k



        mask = torch.tril(
                    torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device)
                )
        # 计算缩放点积注意力
        attn_output = scaled_dot_product_attention(rope_q, rope_k, range_v,mask=mask)  # (batch_size, num_heads, seq_len, d_k)

        range_attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')  # (batch_size, seq_len, d_model)

        out = self.out_proj(range_attn_output)  # (batch_size, seq_len, d_model)

        return out

if __name__ == "__main__":
    # 测试代码
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    print(softmax(x, i=1))  # 沿着最后一维计算 softmax
    print(softmax(x, i=0))  # 沿着第一维计算 softmax
    print(torch.softmax(x, dim=1))  # 使用 PyTorch 内置的 softmax 函数进行对比
    print(torch.softmax(x, dim=0))  # 使用 PyTorch 内置的 softmax 函数进行对比

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
    q = x
    k = x
    v = x
    mask = torch.tensor([[False, False], [True, True]])  # (2, 2)
    print(scaled_dot_product_attention(q, k, v, mask))