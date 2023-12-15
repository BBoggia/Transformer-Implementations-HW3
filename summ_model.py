import torch
import torch.nn as nn
import math
import torch.optim as optim

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # Vector: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even and cos to odd
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pos_encoding', positional_encoding)

    def forward(self, x):
        x = x + (self.pos_encoding[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, features: int,  eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(features)) # Added

    def forward(self, x): #  should be (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiheadAttention(nn.Module):

    def __init__(self, d_model: int, head_count: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # The size of the embedding
        self.head_count = head_count # The number of heads
        assert d_model % head_count == 0, "d_model not divisible by head_count"

        self.d_k = d_model // head_count
        self.weights_query = nn.Linear(d_model, d_model, bias = False)
        self.weights_key = nn.Linear(d_model, d_model, bias = False)
        self.weights_value = nn.Linear(d_model, d_model, bias = False)
        self.weights_output = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        '''
        Applies the attention mechanism to query, key, and value matricies. Returns the attention output and attention scores as a tuple.\n
        Takes a query, key, and value matrix, a mask boolean, and dropout property
        '''
        # The last dimension for the query, key, and value which is the dynamic size of each head
        d_k = query.shape[-1]

        # Results in a matrix of size (batch, head_count, seq_len, seq_len) 
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1) # (batch, head_count, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # (batch, head_count, seq_len, d_k), (batch, head_count, seq_len, seq_len)

    def forward(self, query, key, value, mask):
        query = self.weights_query(query) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.weights_key(key)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.weights_value(value) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, head_count, d_k) --> (batch, head_count, seq_len, d_k)
        # Result is that it splits the query by batches and heads so that each head is seq_len by d_k
        # This ensures that each head has access to the full sequesnce of words but has different embeddings
        query = query.view(query.shape[0], query.shape[1], self.head_count, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head_count, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head_count, self.d_k).transpose(1, 2)

        # (batch, head_count, seq_len, d_k) --> (batch, head_count, seq_len, d_k)
        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, head_count, seq_len, d_k) --> (batch, seq_len, head_count, d_k) --> (batch, seq_len, d_model)
        # The purpose of this is to concatenate all of the heads back together into the origional sequence embedding matrix
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head_count * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # Applies the final output weight matrix to the results from the attention mechanism
        return self.weights_output(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # (batch, seq_len, d_model) + (batch, seq_len, d_model)
    
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention: MultiheadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    # source_mask to apply to the input of the encoder
    # hides interaction of padding word with other words
    def forward(self, x, source_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, source_mask))
        x = self.residual_connection[1](x, self.feed_forward)
        # Combines results of feed forward layer and self attention layer for first skip connection in the encoder
        return x # (batch, seq_len, d_model)
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # (batch, seq_len, d_model)
    

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention: MultiheadAttention, cross_attention: MultiheadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_conn = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    # source_mask = encoder mask, target_mask = decoder mask
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attention(x, x, x, target_mask)) # (batch, seq_len, d_model)
        x = self.residual_conn[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, source_mask)) # (batch, seq_len, d_model)
        x = self.residual_conn[2](x, self.feed_forward) # (batch, seq_len, d_model)

        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x) # (batch, seq_len, d_model)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj_layer(x) # (batch, seq_len, vocab_size)
    
class WarmUpLR:
    def __init__(self, optimizer, warmup_steps=1000):
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

    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding, source_pos: PositionalEncoding, target_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.proj_layer = proj_layer

    def encode(self, source, source_mask):
        source = self.source_embedding(source) # (batch, seq_len, d_model)
        source = self.source_pos(source) # (batch, seq_len, d_model)

        return self.encoder(source, source_mask) # (batch, seq_len, d_model)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target) # (batch, seq_len, d_model)
        target = self.target_pos(target) # (batch, seq_len, d_model)

        return self.decoder(target, encoder_output, source_mask, target_mask) # (batch, seq_len, d_model)
    
    def project(self, x):
        return self.proj_layer(x) # (batch, seq_len, vocab_size)
    

def buildTransformer(source_vocab_size: int, target_vocab_size: int, source_seq_len: int, target_seq_len: int, d_model: int = 512, N: int = 5, head_count: int = 8, learning_rate = 1e-4, dropout: float = 0.1, d_ff: int = 2048, warmup_steps: int = 1000):
    # Build embedding layers
    source_embedding = InputEmbedding(d_model, source_vocab_size)
    target_embedding = InputEmbedding(d_model, target_vocab_size)

    # Build positional embeding layers
    source_pos_embedding = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos_embedding = PositionalEncoding(d_model, target_seq_len, dropout)

    # Build encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoding_self_attention = MultiheadAttention(d_model, head_count, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoding_block = EncoderBlock(d_model, encoding_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoding_block)

    # Build decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiheadAttention(d_model, head_count, dropout)
        decoder_cross_attention =  MultiheadAttention(d_model, head_count, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    # Build Encoder and Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Build Projection Layer
    proj_layer = ProjectionLayer(d_model, target_vocab_size)

    # Build transformer
    transformer = Transformer(encoder, decoder, source_embedding, target_embedding, source_pos_embedding, target_pos_embedding, proj_layer)

    # Return transformer, optimizer, and lr_scheduler
    optimizer:optim.Adam = optim.Adam(transformer.parameters(), lr=learning_rate)
    lr_scheduler: WarmUpLR = WarmUpLR(optimizer, warmup_steps=warmup_steps)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer, optimizer, lr_scheduler