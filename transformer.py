import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self,emb,heads=8):
        """
        emb: dimension of embedding vector
        heads: number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        assert(emb%heads == 0, "embedding dimension must be divisble by number of heads")
        self.emb = emb
        self.heads = heads
        self.heads_dim = emb//heads
        print(self.heads_dim)
        
        self.Wq = nn.Linear(self.heads_dim,self.heads_dim,bias=False)
        self.Wv = nn.Linear(self.heads_dim,self.heads_dim,bias=False)
        self.Wk = nn.Linear(self.heads_dim,self.heads_dim,bias=False)
        
        self.out = nn.Linear(self.heads_dim*heads, self.emb)
        
        
    def forward(self,X):
        """
        X: input data with dimensions (batch, number of embedding vectors, dimension of embedding vectors)
        """
        b, n, d = X.shape
        
        data = X.view(b,n,self.heads,self.heads_dim)
        q = self.Wq(data) #(b,n,heads,heads_dim)
        v = self.Wv(data) #(b,n,heads,heads_dim)
        k = self.Wk(data) #(b,n,heads,heads_dim)
        W = torch.einsum("nqhd,nkhd->nhqk",[q,k])
        attention = F.softmax(W/(d**(1/2)),dim=3)
        Y = torch.einsum("nhqk,nkhd->nqhd",[attention,v]).reshape(b,n,self.emb) #(b,n,d)
        
        return self.out(Y)
    
    
    
    
class Transformer(nn.Module):
    def __init__(self,emb,heads=8,dropout=0.0,mult=4):
        """
        emb: dimension of embedding vector
        heads: number of attention heads
        dropuout: droupout rate
        mult: forward expansion factor
        """
        super(Transformer,self).__init__()
        
        self.attention = MultiHeadAttention(emb,heads)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        nn.fout = nn.Sequential(nn.Linear(emb,emb*mult),
                                nn.ReLU(),
                                nn.Linear(emb*mult,emb))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,X):
        attention = self.attention(X)
        y = self.dropout(self.norm1(attention + X))
        fout = self.fout(y)
        out = self.dropout(self.norm2(fout+y))
        return out
    
    
    