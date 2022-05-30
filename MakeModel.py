import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from MultiHeadedAttention import MultiHeadedAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from PositionalEncoding import PositionalEncoding
from EncoderDecoder import EncoderDecoder

from Encoder import Encoder
from EncoderLayer import EncoderLayer

from Decoder import Decoder
from DecoderLayer import DecoderLayer

from Embeddings import Embeddings
from Generator import Generator

from LabelSmoothing import LabelSmoothing
from NoamOpt import NoamOpt
from SimpleLossCompute import SimpleLossCompute
from Batch import Batch

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    print("what", torch.from_numpy(subsequent_mask) == 0)
    return torch.from_numpy(subsequent_mask) == 0
'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
'''
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
            """https://zhuanlan.zhihu.com/p/74274453
            #權值初始化 Xavier均勻分佈"""
    return model
	
	
#tmp_model = make_model(10, 10, 2)
#print(tmp_model)
def data_gen(V, batch, nbatches): #V=11,batch=30,nbatches=20,
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches): #https://www.zhihu.com/question/316590913
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)).astype(np.int64))
        "返回一個隨機整型數，範圍從低（包括）到高（不包括），即[low, high)。"
        "size=30*10"
        #print(data)		
        data[:, 0] = 1
        #print(data)		
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)   #https://openhome.cc/Gossip/Python/YieldGenerator.html
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        #("SRC",batch.src.shape)
        #print("TRG",batch.src[0])
        #print("batch.trg_mask",batch.src_mask[0])
        out = model(batch.src, batch.trg,  #model.forward
                            batch.src_mask, batch.trg_mask)
        #print("out",out)
        #print("batch.trg_y",batch.trg_y)
        #print("batch.ntokens",batch.ntokens)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
        #print(batch)
    return total_loss / total_tokens

V = 11
"loss"
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2,)
"lr學習率"
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#print(model)
for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    #model.eval()
    #print(run_epoch(data_gen(V, 30, 5), model, 
                    #SimpleLossCompute(model.generator, criterion, None)))
torch.save(model,'./test.pth')
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    print("ys",ys)
    print("ys2",ys.size(1))	
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        #print("out.shape",out.shape) #out.shape torch.Size([1, 1, 512])
        #print("out[:, -1]",out[:, -1]) 		# 512 tensor
        prob = model.generator(out[:, -1])
        #("prob",prob) #prob tensor([[-10.6553,  -1.4391,  -0.7876,  -2.4519,  -8.6053,  -3.4266,  -3.3424, -3.5583,  -3.9485,  -2.4535,  -3.9147]],
        #print("prob.shape",prob.shape) #prob.shape torch.Size([1, 11])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        if next_word == 9 :
            break
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        print("ys3",ys)
    return ys
	
	
model_test = torch.load("./test.pth", map_location='cpu')
"https://mlog.club/article/2862272"
model_test.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,9,8,7,6,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
"http://nlp.seas.harvard.edu/2018/04/03/attention.html"
print("src.size()",src.size())#[1,10]
print("src",src)
print("greedy_decode",greedy_decode(model_test, src, src_mask, max_len=11, start_symbol=1))
"""
https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb
https://huggingface.co/blog/how-to-generate
"""

