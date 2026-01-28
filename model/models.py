import torch.nn as nn
import torch
import warnings
import math
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple





class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, init=None, trainable=True):
        """
        Args:
            vocab_size: size of vocabulary
        """
        super(Embedding, self).__init__()
        if init is not None:
            self.embed = nn.Embedding.from_pretrained(init).requires_grad_(trainable)
        else:
            self.embed = nn.Embedding(vocab_size, d_model).requires_grad_(trainable)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)

        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model, init=None, trainable=True):
        """
        Args:
            max_seq_len: maximium length of input sequence
        Final embedding dimension is max_seq_len + static embedding dimension
        """

        super(PositionalEmbedding, self).__init__()
        if init is not None:
            self.pe = nn.Embedding.from_pretrained(init).requires_grad_(trainable)
        else:
            self.pe = nn.Embedding(max_seq_len, d_model).requires_grad_(trainable)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: output
        """
        # append positional encodings to static embeddings
        seq_len = x.size(1)
        batch_size = x.size(0)
        pos = torch.arange(0, seq_len, dtype=torch.long)
        out = x + self.pe(pos).repeat(batch_size, 1, 1)
        return out
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Rotary embedding helper function"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class MultiHeadAttention(nn.Module):
    def __init__(self, config, linear_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.linear_attn = linear_attn
        self.pos = config.pos
        self.d_k = config.d_model // config.num_heads
        assert (
            config.d_model % config.num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        if config.pos == "relative":
            self.att_bias = nn.Parameter(
                torch.zeros(config.num_heads, config.max_seq_len, config.max_seq_len)
            ).to(config.device)

        if config.pos == "rotary":
            self.freqs_cis = precompute_freqs_cis(
                self.d_model // self.num_heads,
                config.max_seq_len * 2,
                config.rotary_theta,
            ).to(config.device)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        assert mask is not None, "Mask=None is not supported now"

        if self.pos == "rotary":
            T = Q.size(2)
            # expected shape for apply_rotary_emb: (batch_size, max_seq_len, num_head, d_head)
            Q, K = apply_rotary_emb(
                Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=self.freqs_cis[:T]
            )
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)

        QK_vals = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.pos == "relative":
            T = QK_vals.size(2)
            QK_vals = QK_vals + self.att_bias[:, :T, :T].view(1, self.num_heads, T, T)

        if mask is not None:
            if not self.linear_attn:
                attn_scores = QK_vals.masked_fill(mask == 0, -1e9)
                attn_probs = torch.softmax(attn_scores, dim=-1)
            else:
                attn_scores = QK_vals.masked_fill(mask == 0, 0)
                attn_probs = attn_scores
        output = torch.matmul(attn_probs, V)
        return output, (attn_probs, QK_vals)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None, output_attn=False):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        output = (output, attn_probs) if output_attn else output
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))



class TFBlock(nn.Module):
    def __init__(
        self,
        config,
        mlp
    ):

        super().__init__()
        self.vocab_size = config.vocab_size
        self.residual = config.residual
        self.drop = config.dropout
        self.norm = config.norm
        self.ff_dim = config.ff_dim
        self.linear_attn = config.linear_attn
        self.mlp = mlp

        # initiating layers
        self.mha = MultiHeadAttention(config, linear_attn=config.linear_attn)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        if config.mlp:
            self.feed_forward = PositionWiseFeedForward(config.d_model, config.ff_dim)

    def forward(self, x, mask):
        #layer normalization
        out = self.ln_1(x) if self.norm else x
        attn_output, attn_probs = self.mha(out, out, out, mask, output_attn=True)
        out = self.dropout1(attn_output) if self.drop is not None else attn_output
        x = x + out if self.residual else x
        if self.mlp:
            out = self.ln_2(x) if self.norm else x
            out = self.feed_forward(out)
            out = self.dropout2(out)
            x = x + out if self.residual else x
        return x, attn_probs




class TFModel(nn.Module):
    def __init__(self, config):
        super(TFModel, self).__init__()
        self.device = config.device
        self.pad_id = config.pad_id
        self.pos = config.pos
        self.num_layers = config.num_layers
        self.output_norm = config.output_norm
        self.vocab_size = config.vocab_size
        self.embed = Embedding(config.vocab_size, config.d_model)
        if self.pos is None:
            self.pos_embed = PositionalEmbedding(config.max_seq_len, config.d_model)
        #hidden layer without FFN
        self.h_1 = nn.ModuleList([TFBlock(config, True) for i in range(config.num_layers)])
        # hidden layer with FFN (Last hidden layer)
        self.h_2 = TFBlock(config, True)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, src, device = None):
        x = (
            self.embed(src)
            if self.pos in ["relative", "rotary"]
            else self.pos_embed(self.embed(src))
        )

        batch_size, seq_len, _ = x.size()

        pad_mask = (src != self.pad_id)
        causal_mask = torch.tril(
                   torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device)
        )


        
        mask = causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask.unsqueeze(1).unsqueeze(2) 

        out = x

        attn_matrices = []
        for i, (block) in enumerate(self.h_1):
            out, attn_probs = block(out, mask)
            attn_matrices.append(attn_probs)

        out, attn_probs = self.h_2(out, mask)
        attn_matrices.append(attn_probs)

        out = self.ln_f(out) if self.output_norm else out
        out = self.fc(out)
        return out, attn_matrices




def beam_search_inference(
    model: nn.Module,
    input_ids: torch.LongTensor,
    beam_width: int = 5,
    max_length: int = 64,
    eos_token_id: int = None,
    device: str = 'cuda'
) -> torch.LongTensor:
    
    model.eval()
    input_ids = input_ids.to(device)
 
    beams: List[Tuple[torch.LongTensor, float]] = [(input_ids, 0.0)]
    
    with torch.no_grad():
        for _ in range(max_length):
            all_candidates: List[Tuple[torch.LongTensor, float]] = []
            
            for seq, score in beams:
                last_token = seq[0, -1].item()
                
       
                if eos_token_id is not None and last_token == eos_token_id:
                    all_candidates.append((seq, score))
                    continue
       
                logits, _ = model(seq)  # logits shape: [1, seq_len, vocab_size]
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width, dim=-1)
         
                for k in range(beam_width):
                    next_token = topk_tokens[0, k].unsqueeze(0).unsqueeze(0)  # shape [1,1]
                    next_score = score + topk_log_probs[0, k].item()
                    new_seq = torch.cat([seq, next_token], dim=1)  
                    all_candidates.append((new_seq, next_score))
            
       
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            
            if eos_token_id is not None and all(seq[0, -1].item() == eos_token_id for seq, _ in beams):
                break
    

    best_seq = beams[0][0]
    return best_seq








