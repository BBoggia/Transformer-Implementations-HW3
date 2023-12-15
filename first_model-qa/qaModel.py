import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create 0 matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Vector seq_len - 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even, cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        # Register buffer 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        # Normalize across last dim
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Return normalized x
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):

    def __init__(self, d_model: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, dff) -> (batch ,seq_len ,  d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiheadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        # Get martix head size
        self.d_k = d_model // h

        # Q, K, V, and output layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        # Dynamic head size for each
        d_k = q.shape[-1]

        # Get attention scores (batch, h, seq_len, seq_len) 
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) 

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) # Change shape to (batch size, 1, 1, seq_len)
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        # Get attention probability distribution (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores


    def forward(self, q, k, v, mask):
        # Apply linear layers to q, k, and v
        query = self.w_q(q) # (batch, seq_len, d_model)
        key = self.w_k(k)   # ^^
        value = self.w_v(v) # ^^

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # Splits the query by batches and heads so each head is seq_len by d_k
        # Gives each head access to full sequesnce of words but different embeddings
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # Concatenate heads back together into origional sequence embedding matrix
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model)
        # Apply final output weight matrix to results from attention mechanism
        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float, use_norm: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Applies dropout to output of sublayer and adds it to input
        return (x + self.dropout(sublayer(self.norm(x)))) if (self.use_norm) else (x + self.dropout(sublayer(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attn: MultiheadAttention, feed_fwd: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_fwd = feed_fwd
        self.dropout = nn.Dropout(dropout)
        self.residual_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # Mask to apply to encoder input
    # Hides the interaction of padding word with other words
    def forward(self, x, mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residual_conn[1](x, self.feed_fwd)
        # Combines results of feed forward and self attention layer for first skip connection in encoder
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, use_norm: bool = True) -> None:
        super().__init__()
        self.layers = layers
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNormalization()

    def forward(self, x, mask):
        # Pass input through the self attention and feed forward layers
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) if (self.use_norm) else x
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attn: MultiheadAttention, cross_attn: MultiheadAttention, feed_fwd: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_fwd = feed_fwd
        self.residual_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    # e_mask = encoder mask, d_mask = decoder mask
    def forward(self, x, encoder_output, e_mask, d_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn(x, x, x, d_mask))
        x = self.residual_conn[1](x, self.cross_attn(x, encoder_output, encoder_output, e_mask))
        x = self.residual_conn[2](x, self.feed_fwd)

        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, use_norm: bool = True) -> None:
        super().__init__()
        self.layers = layers
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNormalization()

    def forward(self, x, encoder_output, e_mask, d_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, e_mask, d_mask)
        return self.norm(x) if (self.use_norm) else x
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj_layer(x), dim = -1)
    
class WarmUpLR:
    def __init__(self, optimizer, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr = optimizer.defaults['lr']
        self.step_num = 0

    def step(self):
        # Update learning rate
        self.step_num += 1
        # Calculate learning rate
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5)))

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, e_in_embedding: InputEmbedding, d_in_embedding: InputEmbedding, e_pos: PositionalEncoding, d_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.e_in_embedding = e_in_embedding
        self.d_in_embedding = d_in_embedding
        self.e_pos = e_pos
        self.d_pos = d_pos
        self.proj_layer = proj_layer

    def encode(self, src, e_mask):
        src = self.e_in_embedding(src)
        src = self.e_pos(src)

        return self.encoder(src, e_mask)

    def decode(self, src, target, e_mask, d_mask):
        target = self.d_in_embedding(target)
        target = self.d_pos(target)

        return self.decoder(target, src, e_mask, d_mask)
    
    def project(self, x):

        return self.proj_layer(x)
    

def buildTransofermer(e_vocab_size: int, d_vocab_size: int, e_seq_len: int, d_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.08, d_ff: int = 2048, warmup_steps: int = 2000):
    # Build embedding layers
    e_in_embedding = InputEmbedding(d_model, e_vocab_size)
    d_in_embedding = InputEmbedding(d_model, d_vocab_size)

    # Build positional embeding layers
    e_pos_embedding = PositionalEncoding(d_model, e_seq_len, dropout)
    d_pos_embedding = PositionalEncoding(d_model, d_seq_len, dropout)

    # Build encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attn = MultiheadAttention(d_model, h, dropout)
        feed_fwd = FeedForward(d_model, d_ff, dropout)
        block = EncoderBlock(self_attn, feed_fwd, dropout)
        encoder_blocks.append(block)

    # Build decoder blocks
    decoder_blocks = []
    for _ in range(N):
        d_self_attn = MultiheadAttention(d_model, h, dropout)
        d_cross_attn =  MultiheadAttention(d_model, h, dropout)
        feed_fwd = FeedForward(d_model, d_ff, dropout)
        d_block = DecoderBlock(d_self_attn, d_cross_attn, feed_fwd, dropout)
        decoder_blocks.append(d_block)

    # Build Encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Build Projection Layer
    proj_layer = ProjectionLayer(d_model, d_vocab_size)

    # Build transformer
    transformer = Transformer(encoder, decoder, e_in_embedding, d_in_embedding, e_pos_embedding, d_pos_embedding, proj_layer)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Return transformer, optimizer, and lr_scheduler
    optimizer = optim.Adam(transformer.parameters(), lr=5e-5)
    lr_scheduler = WarmUpLR(optimizer, warmup_steps=warmup_steps)

    return transformer, optimizer, lr_scheduler