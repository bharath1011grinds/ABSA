import json
import torch
from pathlib import Path

from model import ABSAEncoder, ABSAClassifier
from data import ABSADataset, parse_semeval_xml, save_json, collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


from config import EMBEDDING_DIM, HIDDEN_DIM, LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = Path("data/raw")
PROCESSED_TEST_DIR = Path("data/processed")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)    


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentence_ids, aspect_ids, labels, _, _ in loader:
            sentence_ids = sentence_ids.to(DEVICE)
            aspect_ids = aspect_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            mask = (sentence_ids != 0).long()

            logits = model(sentence_ids, aspect_ids, mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, macro_f1


if __name__ == '__main__':

    val_data = load_json(PROCESSED_TEST_DIR/"validation.json")
    vocab = load_json(PROCESSED_TEST_DIR/"vocab.json")

    test_ds = ABSADataset(val_data, vocab, LABEL_MAP)

    test_loader = DataLoader(test_ds, 32, shuffle=False, collate_fn=collate_fn)

    encoder = ABSAEncoder(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=vocab["<PAD>"]
    )

    model = ABSAClassifier(encoder, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=DEVICE))
    model.to(DEVICE)

    acc, macro_f1 = evaluate(model, test_loader)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}") 









