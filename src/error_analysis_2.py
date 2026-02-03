import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from eval import load_json, evaluate
from data import tokenize

from data import ABSADataset, collate_fn
from model import ABSAClassifier, ABSAEncoder, BiLSTMEncoder
from config import LABEL_MAP
from config import EMBEDDING_DIM, HIDDEN_DIM, LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = Path("data/raw")
PROCESSED_TEST_DIR = Path("data/processed")


POS_WORDS = {
    "good", "great", "excellent", "amazing", "nice", "love",
    "loved", "awesome", "perfect", "best", "wonderful"
}

NEG_WORDS = {
    "bad", "terrible", "awful", "poor", "worst", "hate",
    "hated", "slow", "disappointing", "disappointed"
}


def find_opinion_positions(tokens, pos_words, neg_words):
    opinion_positions = []
    for i, token in enumerate(tokens):
        if token in pos_words or token in neg_words:
            opinion_positions.append(i)
    return opinion_positions


def find_aspect_span(sentence_tokens, aspect_tokens):
    aspect_len = len(aspect_tokens)
    for i in range(len(sentence_tokens) - aspect_len + 1):
        if sentence_tokens[i:i+aspect_len] == aspect_tokens:
            return i, i + aspect_len - 1
    return None


def compute_min_distance(op_positions, aspect_pos):
    start, end = aspect_pos
    
    distances = [min(abs(op-start), abs(op-end)) for op in op_positions]

    return min(distances)

def opinion_distance(sentence_tokens, aspect_tokens):


    aspect_span = find_aspect_span(sentence_tokens, aspect_tokens)
    if aspect_span is None:
        return None

    opinion_positions = find_opinion_positions(sentence_tokens, POS_WORDS, NEG_WORDS)
    if not opinion_positions:
        return None

    return compute_min_distance(opinion_positions, aspect_span)


def bucket_distance(d):
    if d <= 2:
        return "near"
    elif d <= 5:
        return "medium"
    else:
        return "far"
    

if __name__ == '__main__':
    val_data = load_json(PROCESSED_TEST_DIR/"validation.json")
    vocab = load_json(PROCESSED_TEST_DIR/"vocab.json")
    near_data = []
    far_data = []
    med_data = []
    none_data = []

    for data in val_data:
        dist = opinion_distance(tokenize(data['sentence']), tokenize(data['aspect']))

        if dist is None: 
            none_data.append(data)
        else:    
            d = bucket_distance(dist)

            if d == 'far':
                far_data.append(data)
            elif d == 'medium':
                med_data.append(data)
            else:
                near_data.append(data)
        

    far_test = ABSADataset(far_data, vocab, LABEL_MAP)
    near_test = ABSADataset(near_data, vocab, LABEL_MAP)
    none_test = ABSADataset(none_data, vocab, LABEL_MAP)
    med_test = ABSADataset(med_data, vocab, LABEL_MAP)


    far_loader = DataLoader(far_test, 32, shuffle=False, collate_fn=collate_fn)
    near_loader = DataLoader(near_test, 32, shuffle=False, collate_fn=collate_fn)
    none_loader = DataLoader(none_test, 32, shuffle=False, collate_fn=collate_fn)
    med_loader = DataLoader(med_test, 32, shuffle=False, collate_fn=collate_fn)

    encoder = ABSAEncoder(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=vocab["<PAD>"]
    )

    model = ABSAClassifier(encoder, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=DEVICE))
    model.to(DEVICE)

    far_acc, far_macro_f1, cm_1 = evaluate(model, far_loader)
    near_acc, near_macro_f1, cm_2 = evaluate(model, near_loader)
    med_acc, med_macro_f1, cm_3 = evaluate(model, med_loader)
    none_acc, none_macro_f1, cm_4 = evaluate(model, none_loader)

    print(f"Confusion matrix for Far Data : {cm_1}")
    print(f"Confusion matrix for Near Data : {cm_2}")
    print(f"Confusion matrix for Medium Data : {cm_3}")
    print(f"Confusion matrix for None Data : {cm_4}")

    print(".......................................................")
    print (f"Far data samples : {len(far_data)}")
    print(f"Near data samples : {len(near_data)}")
    print(f"Medium data samples : {len(med_data)}")
    print(f"None Data samples : {len(none_data)}")
'''
    print(f"Far Data Accuracy : {far_acc:.4f}","   ",f"Far Macro F1 : {far_macro_f1:.4f}")
    print(f"Near Data Accuracy      : {near_acc:.4f}","   ",f"Near Macro F1 : {near_macro_f1:.4f}") 
    print(f"Medium Data Accuracy : {med_acc:.4f}","   ",f"Medium Macro F1 : {med_macro_f1:.4f}")
    print(f"None Data Accuracy      : {none_acc:.4f}","   ",f"None Macro F1 : {none_macro_f1:.4f}") '''













