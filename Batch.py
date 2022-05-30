import numpy as np
import torch
from torch.autograd import Variable

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        #print("src",src.shape)
        #print((src != pad))
        self.src_mask = (src != pad).unsqueeze(-2)
        #print("trg",trg.shape)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        #print("trg_mask.shape",self.trg_mask.shape)          
        #print("self.trg_y",self.trg_y)
        #print("self.ntokens",self.ntokens)		
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        #print("tgt",tgt.shape)  
        tgt_mask = (tgt != pad).unsqueeze(-2)
        #print("tgt_mask",tgt_mask.shape)  
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))			
        return tgt_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0