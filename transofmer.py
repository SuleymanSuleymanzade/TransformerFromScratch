import torch 
import torch.nn as nn 
import math 

class EmbeddingLayer:
    def __init__(self, d_model:int, vocab_size:int):
        self.d_model = d_model
        self.vocab_size = vocab_size 
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
    
class PositionalEncoding:
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        assert 0 <= dropout <= 1

        self.d_model = d_model 
        self.seq_len = seq_len 

        pe:torch.Tensor = torch.zeros(self.seq_len, self.d_model)
        position:torch.Tensor = torch.arange(0, self.seq_len, dtype=torch.float32).unsqueeze(1)

        div_term:torch.Tensor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)





    