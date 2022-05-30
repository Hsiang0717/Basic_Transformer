import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)#512 , 11 => 0~10

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
#prob tensor([[-10.6553,  -1.4391,  -0.7876,  -2.4519,  -8.6053,  -3.4266,  -3.3424, -3.5583,  -3.9485,  -2.4535,  -3.9147]],