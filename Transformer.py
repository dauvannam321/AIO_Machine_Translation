import math

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop_out=0.1):
        super().__init__()
        self.inf = 1e9

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, self.d_model) # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values

class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, drop_out=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x

class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model, device):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, drop_out)
        self.drop_out_1 = nn.Dropout(drop_out)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, drop_out)
        self.drop_out_2 = nn.Dropout(drop_out)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        ) # (B, L, d_model)

        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, num_heads, drop_out)
        self.drop_out_1 = nn.Dropout(drop_out)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, drop_out)
        self.drop_out_2 = nn.Dropout(drop_out)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, drop_out)
        self.drop_out_3 = nn.Dropout(drop_out)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, drop_out=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, drop_out) for i in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, drop_out):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, drop_out) for i in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.src_embedding = nn.Embedding(self.cfg.sp_vocab_size, self.cfg.d_model)
        self.tgt_embedding = nn.Embedding(self.cfg.sp_vocab_size, self.cfg.d_model)
        self.positional_encoder = PositionalEncoder(
            self.cfg.seq_len,
            self.cfg.d_model,
            self.cfg.device
        )
        self.encoder = Encoder(
            self.cfg.num_layers,
            self.cfg.d_model,
            self.cfg.num_heads,
            self.cfg.d_ff,
            self.cfg.drop_out
        )
        self.decoder = Decoder(
            self.cfg.num_layers,
            self.cfg.d_model,
            self.cfg.num_heads,
            self.cfg.d_ff,
            self.cfg.drop_out
        )
        self.output_linear = nn.Linear(self.cfg.d_model, self.cfg.sp_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, tgt_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        tgt_input = self.tgt_embedding(tgt_input) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        tgt_input = self.positional_encoder(tgt_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(tgt_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output