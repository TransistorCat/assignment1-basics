import torch
import torch.nn as nn
from .MHA import MultiHeadSelfAttention
from .FFN import FFN
from .RMSnorm import RMSNorm
from .linear import Linear
from .Rope import RotaryPositionalEmbedding
from .embedding import Embedding
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, rope_theta: float) -> None:
        super().__init__()
        '''
        d_model: int  Transformer 块输入维度
        num_heads: int  注意力头数
        d_ff: int  前馈网络中间层维度
        max_seq_len: int  RoPE 最大序列长度
        rope_theta: float  RoPE 基数频率
        '''
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)


        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        输入形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, seq_len, d_model)
        '''
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float
    ) -> None:
        super().__init__()
        '''
        d_model: int  模型隐藏层维度
        num_heads: int  注意力头数
        d_ff: int  前馈网络中间层维度
        vocab_size: int  词表大小
        context_length: int  最大上下文长度
        num_layers: int  Transformer Block 层数
        rope_theta: float  RoPE 基数频率
        '''

        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.lm_head = Linear(d_model, vocab_size)
        self.norm = RMSNorm(d_model)
        pass

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        tokens: (batch_size, seq_len) 整数 token IDs
        return: (batch_size, seq_len, vocab_size) next-token logits
        '''
        x = self.token_embedding(tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.lm_head(x)

        return x