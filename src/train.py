import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ABSADataset, collate_fn
from model import ABSAClassifier, ABSAEncoder, BiLSTMEncoder
from config import LABEL_MAP


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(p):
    with open(p,  'r', encoding='utf-8') as f:
        return json.load(f)
    
train_data = load_json('data/processed/train.json')
vocab = load_json('data/processed/vocab.json')

train_ds = ABSADataset(train_data, vocab, LABEL_MAP)
train_loader = DataLoader(train_ds, batch_size=32, shuffle= True,collate_fn= collate_fn)


embedding_dim = 300
hidden_dim = 100

encoder = ABSAEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(vocab),
                      padding_idx=vocab["<PAD>"])

model = ABSAClassifier(encoder, hidden_dim=hidden_dim).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) #using adaptive moments as the optimizer


def train_one_epoch(model, loader):

    model.train() #puts the model in train mode - dropout is active, dropout is off, when we do, model.eval()
    total_loss = 0.0

    for sentence_ids, aspect_ids, labels, sent_lens, _ in loader:
        
        sentence_ids = sentence_ids.to(DEVICE)
        aspect_ids = aspect_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        mask = (sentence_ids!=0).long() #1 if non zero id, 0 if the word has id as zero. masks the pads

        optimizer.zero_grad() #Resets the accumulated gradients to zero, every batch starts afresh.
        logits = model(sentence_ids, aspect_ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

EPOCHS = 10

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader) #average loss in that epoch
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
    
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model.pt")
print("Model saved.")


