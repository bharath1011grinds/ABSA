import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from eval import load_json, evaluate

from data import ABSADataset, collate_fn
from model import ABSAClassifier, ABSAEncoder, BiLSTMEncoder
from config import LABEL_MAP
from config import EMBEDDING_DIM, HIDDEN_DIM, LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = Path("data/raw")
PROCESSED_TEST_DIR = Path("data/processed")





if __name__ == '__main__':
    val_data = load_json(PROCESSED_TEST_DIR/"validation.json")
    vocab = load_json(PROCESSED_TEST_DIR/"vocab.json")
    long_data = []
    short_data = []

    for data in val_data:
        if len(data['aspect'].split(' '))>1:
            long_data.append(data)
        else: 
            short_data.append(data)


    long_test = ABSADataset(long_data, vocab, LABEL_MAP)
    short_test = ABSADataset(short_data, vocab, LABEL_MAP)

    long_loader = DataLoader(long_test, 32, shuffle=False, collate_fn=collate_fn)
    short_loader = DataLoader(short_test, 32, shuffle=False, collate_fn=collate_fn)

    encoder = ABSAEncoder(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=vocab["<PAD>"]
    )

    model = ABSAClassifier(encoder, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=DEVICE))
    model.to(DEVICE)

    long_acc, long_macro_f1 = evaluate(model, long_loader)
    short_acc, short_macro_f1 = evaluate(model, short_loader)

    print(f"Long Data Accuracy : {long_acc:.4f}","   ",f"Short Data Accuracy : {short_acc:.4f}")
    print(f"Long Macro F1      : {long_macro_f1:.4f}","   ",f"Short Macro F1 : {short_macro_f1:.4f}") 
