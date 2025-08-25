import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value shape: (seq_len, batch, embed_dim)
        attn_output, _ = self.mha(query, key, value, attn_mask=attn_mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_output = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, attn_mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch)
        x = self.embedding(src)  # (seq_len, batch, embed_dim)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, num_heads, ff_dim, num_layers, max_len=100):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, num_heads, ff_dim, num_layers, max_len)
        self.decoder = Decoder(tgt_vocab, embed_dim, num_heads, ff_dim, num_layers, max_len)
        self.output_layer = nn.Linear(embed_dim, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, memory_mask)
        out = self.output_layer(dec_out)
        return out


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



src_vocab_size = 5000
tgt_vocab_size = 5000
embed_dim = 128
num_heads = 8
ff_dim = 512
num_layers = 2
max_len = 50
batch_size = 4
seq_len = 20

model = Transformer(src_vocab_size, tgt_vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(3):

    src = torch.randint(1, src_vocab_size, (seq_len, batch_size)).to(device)
    tgt = torch.randint(1, tgt_vocab_size, (seq_len, batch_size)).to(device)

    tgt_input = tgt[:-1, :]
    tgt_output = tgt[1:, :]

    tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

    optimizer.zero_grad()
    output = model(src, tgt_input, tgt_mask=tgt_mask)

    loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
