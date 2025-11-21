import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F 




class CBOW_MODEL(Module): 
    def __init__(self, vocab_size, embedding_size, context_size):  #context size is how many words around the the center word we're predicting
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size) #create embedding layer 
        self.fc1 = nn.Linear(embedding_size* context_size, 128) #check why we do embedding size * context size 
        self.fc2 = nn.Linear(128, vocab_size) #project to a word within the vocab size
    
    def forward(self, input):  
        input = self.embedding_layer(input) #after doing this, our input will have shape: (batch, context_size, d_model), we need to flatten it before passing into fc1
        input = input.view(input.size(0), -1) #flattens the embedding and context size. New tensor is (batch_size, embedding_size * context size). Note that this linear layer is taking a
        #two dimensional output, but it's okay. The linear transformation is perforemd on the last dimension of the tensor (in this case, embedding_size * context_size)
        input = self.fc1(input)
        input = F.relu(input)
        logits = self.fc2(input)

        logits = F.log_softmax(logits, dim=1)

        #input = F.softmax(input, dim=-1) already incorporated in loss 
        return logits
