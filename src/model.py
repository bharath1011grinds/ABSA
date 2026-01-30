import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, embedding_dim, padding_idx = 0):
        super().__init__()

        self.embedding = nn.Embedding(embedding_dim=embedding_dim, padding_idx=padding_idx,
                                      num_embeddings=vocab_size)
        
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                              bidirectional=True, batch_first=True)
        
    def forward(self, sentence_ids, sentence_lens=None):

        #Batchsize*Timesteps*Embedding_dim
        embeddings = self.embedding(sentence_ids)#internally calls nn.Embedding.forward()

        #Batchsize * Timesteps * (2*Hidden_Dim) 2x cos representation is bi-directional.
        outputs,_ = self.bilstm(embeddings)

        return outputs
    
class AspectAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W_h = nn.Linear(2*hidden_dim, 2*hidden_dim)#Bias is true here because, the bias acts on the sammples before the tanh activation and adding bias will help in activating or supressing the feature depending on the threshold.
        self.W_a = nn.Linear(2*hidden_dim, 2*hidden_dim)#Bias reasoning same as above.
        self.v = nn.Linear(2*hidden_dim, 1, bias= False) #Bias is false here, because same bias would be applied to all the samples, post tanh activation and the softmax would cancel the bias beacause its a constant term added to all the scores, thereby the end probabilities remain unchanged.


    def forward(self, H, aspect_vector, mask=None):
        """
        H: [B, T, 2H]   (BiLSTM outputs)
        aspect_vec: [B, 2H]
        mask: [B, T] (1 for real tokens, 0 for PAD)
        """

        #Expanding the aspect vector along the timestep dimension, to make it comparable to H
        aspect_expanded = aspect_vector.unsqueeze(1).expand_as(H)

        #Calculate Attention scores
        scores = self.v(torch.tanh(self.W_h(H)+self.W_a(aspect_expanded))).squeeze(-1) #scores -> B*T

        if mask is not None:

            scores = scores.masked_fill(mask == 0, -1e9) #assigning a very large negative number if we need that term to be masked.

        alpha = torch.softmax(scores, dim=1) #B*T


        weighted_sum = torch.sum(H*alpha.unsqueeze(-1), dim=1) #NOTE: The multiplication here is element-wise mul and it does not change the shape of the matrix, only the sum alters the shape.
        #The above matrix will have a shape B*2H, the time dimension gets compressed because of the sum along that dimension

        return weighted_sum


class ABSAEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, padding_idx=0):

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                      padding_idx=padding_idx )
        
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.attention = AspectAttention(hidden_dim)#initialising an AspectAttention object.

        self.aspect_proj = nn.Linear(embedding_dim, 2*hidden_dim)

    def forward(self, sentence_ids, aspect_ids, sentence_mask = None):

        #sentence embeddings
        embedded = self.embedding(sentence_ids) #B*T*E

        #bilstm contextual embeddings
        H,_ = self.bilstm(embedded) #B*T*2H


        #aspect embedding
        asp_emb = self.embedding(aspect_ids) #B*A*E
        asp_emb = asp_emb.mean(dim=1) #B*E
        asp_vec = self.aspect_proj(asp_emb)#Projecting the mean aspect embedding be compatible with the sentence embedding.

        out = self.attention(H, asp_vec, sentence_mask)

        return out
    
class ABSAClassifier(nn.Module):
    
    def __init__(self, encoder, hidden_dim, num_classes=3, dropout=0.3):
        
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_dim, num_classes)


    def forward(self, sentence_ids, aspect_ids, sentence_mask=None):

        rep = self.encoder(sentence_ids, aspect_ids, sentence_mask) #Passes to ABSAEncoder() and we get a B*2HH representtation with attention integrated.
        rep = self.dropout(rep)#Drops neurons with a prob of 0.3
        logits = self.fc(rep)#Returns a B*3 output

        return logits
    



        



                
        