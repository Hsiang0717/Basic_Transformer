import torch.nn as nn
import math, copy, time

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        "個數,維度"
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        #print("1 Embeddings : "x)
        #print("2 Embeddings : ",self.lut(x).size())
        #print("3 Embeddings : ",(self.lut(x) * math.sqrt(self.d_model)).size())	
        #print("4 Embeddings : ",math.sqrt(self.d_model),self.d_model)
        #print("5 Embeddings : ",self.lut(x) * math.sqrt(self.d_model))			
        return self.lut(x) * math.sqrt(self.d_model)
        #print("5 Embeddings : ",x)	        
        #return x		