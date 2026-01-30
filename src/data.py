import json
from pathlib import Path
from lxml import etree
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

#Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

TRAIN_FILE = RAW_DATA_DIR / "Restaurants_Train_v2.xml"
TEST_FILE = RAW_DATA_DIR/"restaurants-trial.xml"

def parse_semeval_xml(file_path):

    #will return a list of (sentence, aspect, polarity) dictionaries

    tree = etree.parse(str(file_path))
    root=tree.getroot()

    samples = []

    for sentence in root.iter('sentence'):
        text_elem = sentence.find('text')
        if text_elem is None:
            continue

        sentence_text=text_elem.text.strip()

        aspect_terms=sentence.find('aspectTerms')
        if aspect_terms is None:
            continue

        for aspect in aspect_terms.iter('aspectTerm'):
            polarity = aspect.get('polarity')
            term = aspect.get('term')

            if term is None or polarity == 'conflict':
                continue

            samples.append({'sentence': sentence_text, 'polarity': polarity, 'aspect': term})

    return samples

def save_json(data, out_path):
    out_path.parent.mkdir(parents = True, exist_ok = True)

    with open(out_path, 'w', encoding = "utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii= False)


def tokenize(text):
    
    text = text.lower()
    #The below RE was added later to remove unwanted punctuations that would increase the Vocab size and add noise, although minimal difference in model convergence, this takes up excess vocab and its better to optimize early on.
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def build_vocab(samples, min_freq=1):
    
    vocab={}
    freq={} #this freq dict is used to have more control over the min_freq, we might have noise like typos that occur just once in the vocab. We can choose to get rid of them if needed.

    for sample in samples: 

        for token in tokenize(sample['sentence']):
            freq[token] = freq.get(token, 0)+1

        for token in tokenize(sample['aspect']):
            freq[token] = freq.get(token, 0)+1
    
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    for token, freq in freq.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab



class ABSADataset(Dataset):

    def __init__(self, samples, vocab, label_map):
        super().__init__()

        self.samples=samples
        self.vocab=vocab
        self.label_map=label_map

    def encode(self, tokens):
        #the below would return a list of integers that correspond to the tokens in the vocabulary.
        return [ self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

#We NEED to provide the definitions of __len__ and __getitem__ fns if we inherit the dataset module. This is for the dataloader to act upon our data.
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        sentence_tokens = tokenize(sample['sentence'])
        aspect_tokens = tokenize(sample['aspect'])

        sentence_ids = self.encode(sentence_tokens)
        aspect_ids = self.encode(aspect_tokens)
        label = self.label_map[sample['polarity']]

        return (torch.tensor(sentence_ids, dtype=torch.long),
                torch.tensor(aspect_ids, dtype=torch.long),
                torch.tensor(label, dtype = torch.long))
    
#Padding logic below:    
def collate_fn(batch):

    sentences, aspects, labels = zip(*batch)

    sentence_lens = [len(s) for s in sentences]
    aspect_lens = [len(a) for a in aspects]

    max_sentence_len = max(sentence_lens)
    max_aspect_len = max(aspect_lens)

    padded_sentences=[]
    padded_aspects=[]

    for s,a in zip(sentences, aspects):
        #print(s, sentences)
        s_padded = torch.cat([s, torch.zeros(max_sentence_len-len(s), dtype=torch.long)])
        a_padded = torch.cat([a, torch.zeros(max_aspect_len-len(a), dtype=torch.long)])

        padded_sentences.append(s_padded)
        padded_aspects.append(a_padded)

    return (torch.stack(padded_sentences),
            torch.stack(padded_aspects),
            torch.tensor(labels),
            torch.tensor(sentence_lens),#we return the sentence and aspect lens cos we wil need them for masking the pads(for attention) and attention normalization,
            torch.tensor(aspect_lens))



    


    
        







if __name__ == "__main__":
    train_data = parse_semeval_xml(TRAIN_FILE)
    print(f"Train samples: {len(train_data)}")

    train_samples, val_samples = train_test_split(train_data, train_size=0.8, test_size=0.2, random_state=42,
                                                  stratify= [s['polarity'] for s in train_data])#used stratify to make sure the sentiment propotions are same in the train and validation data


    save_json(train_samples, PROCESSED_DATA_DIR/"train.json")
    save_json(val_samples, PROCESSED_DATA_DIR/"validation.json")
    print(f"sample example: {train_data[0]}")


    vocab = build_vocab(train_samples)
    save_json(vocab, PROCESSED_DATA_DIR/"vocab.json")
    print("vocab size:", len(vocab))


