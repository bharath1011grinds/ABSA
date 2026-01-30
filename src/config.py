LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()} #Inverting the (key, value) pairs in the above dict

EMBEDDING_DIM = 300
HIDDEN_DIM = 100

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"