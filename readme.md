Overview:
This project implements an end-to-end Aspect-Based Sentiment Analysis (ABSA) system using PyTorch, built completely from scratch without pretrained language models.

The goal is to predict the sentiment polarity (positive / neutral / negative) of a sentence with respect to a specific aspect, rather than the sentence as a whole.

E.g.
Sentence: “The food was great, but the service was slow.”
Aspect: service
Prediction: negative

Dataset used: SemEval 2014 – Task 4 (Restaurants domain)

Important note on evaluation:
The official SemEval test labels are not publicly available.
Therefore, evaluation is performed using an 80/20 stratified split of the training data, which is a standard and accepted practice.


Key components:
Embedding layer (randomly initialized, trainable)
BiLSTM for contextual word representations
Aspect-aware additive attention
Linear classifier for sentiment prediction

The attention mechanism allows the model to focus on aspect-relevant tokens rather than treating the sentence uniformly.


Aspect aware attention: et = V@tanh(W_h.ht+W_a.a)

Training Details:

Optimizer: Adam
Loss: CrossEntropyLoss
Batch size: 32
Embedding dimension: 300
Hidden dimension: 100
Training from scratch (no pretrained embeddings)


Results: 
Accuracy score - 71.15%
Macro F1 - 62.21

Steps to run: 

Preprocessing - python src/data.py
Training - python src/train.py
Evaluation - python src/eval.py
Prediction - python src/predict.py
