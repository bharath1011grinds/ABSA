import json
import torch

from model import ABSAEncoder, ABSAClassifier
from data import tokenize

from config import ID2LABEL, EMBEDDING_DIM, HIDDEN_DIM

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


def load_vocab(path="data/processed/vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def encode(tokens, vocab):
    return [vocab.get(token, vocab.get("<UNK>")) for token in tokens]#returns a list of integers

def predict(sentence, aspect, model, vocab):
    model.eval() #switching the model to eval mode, dropout is inactive.

    sent_tokens = tokenize(sentence)
    aspect_tokens = tokenize(aspect)

    sentence_ids = torch.tensor([encode(sent_tokens, vocab)], dtype=torch.long).to(DEVICE)
    aspect_ids = torch.tensor([encode(aspect_tokens, vocab)], dtype=torch.long).to(DEVICE)

    mask =(sentence_ids!=vocab["<PAD>"]).long()

    with torch.no_grad():
        logits = model(sentence_ids, aspect_ids, mask)
        pred_id = torch.argmax(logits, dim=-1).item()#returning argmax because, higher the logit value, higher the probability to occur. Model thinks the higher prob label has a high chance.
    
    return ID2LABEL[pred_id]



if __name__ == '__main__':
    vocab = load_vocab()
   
    encoder = ABSAEncoder(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=vocab["<PAD>"]
    )

    model = ABSAClassifier(encoder, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=DEVICE))
    model.to(DEVICE)

    sentence = "I loved the ambience, although the prices were a bit too high."
    aspect = "ambience"
    prediction = predict(sentence, aspect, model, vocab)

    print(f"The sentiment for the aspect '{aspect}' is {prediction}")



