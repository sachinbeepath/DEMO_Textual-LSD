import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads=8):
        """
        emb: dimension of embedding vector
        heads: number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        assert emb % heads == 0, "embedding dimension must be divisble by number of heads"
        self.emb = emb
        self.heads = heads
        self.heads_dim = emb // heads

        self.queries = nn.Linear(self.emb, self.emb, bias=False)
        self.keys = nn.Linear(self.emb, self.emb, bias=False)
        self.values = nn.Linear(self.emb, self.emb, bias=False)
        self.Wq = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.Wv = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.Wk = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        self.out = nn.Linear(self.heads_dim * heads, self.emb)

    def forward(self, X):
        """
        X: input data with dimensions (batch, number of embedding vectors, dimension of embedding vectors)
        """
        b, n, d = X.shape
        # 2x102x128
        # 2x102x4x32
        queries = self.queries(X)
        keys = self.keys(X)
        values = self.values(X)
        # data = X.view(b,n,self.heads,self.heads_dim)
        queries = queries.reshape(b, n, self.heads, self.heads_dim)
        keys = keys.reshape(b, n, self.heads, self.heads_dim)
        values = values.reshape(b, n, self.heads, self.heads_dim)

        q = self.Wq(queries)  # (b,n,heads,heads_dim)
        v = self.Wv(values)  # (b,n,heads,heads_dim)
        k = self.Wk(keys)  # (b,n,heads,heads_dim)
        W = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        attention = F.softmax(W / (d ** (1 / 2)), dim=3)
        Y = torch.einsum("nhqk,nkhd->nqhd", [attention, v]).reshape(b, n, self.emb)  # (b,n,d)

        return self.out(Y)


class Transformer(nn.Module):
    def __init__(self, emb, heads=8, dropout=0.0, mult=4):
        """
        emb: dimension of embedding vector
        heads: number of attention heads
        dropuout: droupout rate
        mult: forward expansion factor
        """
        super(Transformer, self).__init__()

        self.attention = MultiHeadAttention(emb, heads)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.fout = nn.Sequential(nn.Linear(emb, emb * mult),
                                  nn.ReLU(),
                                  nn.Linear(emb * mult, emb))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        attention = self.attention(X)
        y = self.dropout(self.norm1(attention + X))
        fout = self.fout(y)
        out = self.dropout(self.norm2(fout + y))
        return out

## encoder to use word embeddings
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict(weights_matrix)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

class Encoder(nn.Module):
    # creating the encoding cell that takes in the transformer
    def __init__(self, vocab_size, emb_size, num_layers, heads, mult, dropout, maxlength, device, weights_matrix):
        """

        Parameters
        ----------
        vocab_size: the size of the dictionary being inputted
        emb_size: the size of each embedding vector
        num_layers
        heads: number of heads
        mult: the forward expansion factor
        dropout:
        maxlength: relates to the positional embedding, the max length of a sentence we want to include (will depend on our data)
        """
        self.device = device
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        # self.device = device
        self.word_emb = create_emb_layer(weights_matrix, True)
        self.positional_emb = nn.Embedding(maxlength, emb_size)

        self.layers = nn.ModuleList([Transformer(emb_size, heads, dropout, mult) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # this will learn how words are structured
        out = self.dropout(self.word_emb(x) + self.positional_emb(positions))
        for layer in self.layers:
            # the out out out refer to the value key and query
            out = layer(out)  # all inputs are the same

        return out








