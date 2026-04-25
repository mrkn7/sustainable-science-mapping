"""Standalone inference helper for the Attention-GRU WOS model.

Usage:
    from models.attention_gru.inference import predict_abstract
    class_id, confidence = predict_abstract("Your abstract here...")
"""
import os
import json
import re

import torch

from models.attention_gru.attention_gru import GRUAttentionEncoder, Classifier

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, "attention_gru_wos.pth")
_VOCAB_PATH = os.path.join(_THIS_DIR, "word2idx.json")
_LABELS_PATH = os.path.join(_THIS_DIR, "labels.json")

_MAX_LEN = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

idx2label = {}
if os.path.exists(_LABELS_PATH):
    with open(_LABELS_PATH, "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}

checkpoint = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
hp = checkpoint["hyperparameters"]

encoder = GRUAttentionEncoder(
    vocab_size=hp["vocab_size"],
    embed_dim=hp["embed_dim"],
    hidden_dim=hp["hidden_dim"],
    num_layers=hp["num_layers"],
    bidirectional=hp["bidirectional"],
).to(device)

clf_input_dim = hp["hidden_dim"] * (2 if hp["bidirectional"] else 1)
classifier = Classifier(
    input_dim=clf_input_dim,
    num_classes=hp["num_classes"],
    dropout=hp["fc_dropout"],
).to(device)

encoder.load_state_dict(checkpoint["encoder_state_dict"])
classifier.load_state_dict(checkpoint["classifier_state_dict"])
encoder.eval()
classifier.eval()


def _tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def predict_abstract(text: str):
    """Return (class_id, confidence) for a given abstract."""
    unk = word2idx["<UNK>"]
    ids = [word2idx.get(t, unk) for t in _tokenize(text)]
    if len(ids) < _MAX_LEN:
        ids = ids + [0] * (_MAX_LEN - len(ids))
    else:
        ids = ids[:_MAX_LEN]

    tensor_input = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        feats = encoder(tensor_input)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

    return pred_class, confidence


def predict_abstract_topk(text: str, k: int = 5):
    """Return list of (class_id, label, confidence) for the top-k predictions."""
    unk = word2idx["<UNK>"]
    ids = [word2idx.get(t, unk) for t in _tokenize(text)]
    if len(ids) < _MAX_LEN:
        ids = ids + [0] * (_MAX_LEN - len(ids))
    else:
        ids = ids[:_MAX_LEN]

    tensor_input = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        feats = encoder(tensor_input)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(k, probs.shape[0])
        top_conf, top_idx = torch.topk(probs, k=k)

    return [
        (int(i.item()),
         idx2label.get(int(i.item()), f"Class {int(i.item())}"),
         float(c.item()))
        for c, i in zip(top_conf, top_idx)
    ]
