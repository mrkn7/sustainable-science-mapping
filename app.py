import os
import json
import re

import torch
import torch.nn as nn
import gradio as gr


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "attention_gru")
MODEL_PATH = os.path.join(MODEL_DIR, "attention_gru_wos.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "word2idx.json")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

MAX_LEN = 250
TOP_K = 5


# MODEL DEFINITIONS 

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs):
        attn_weights = torch.softmax(self.attention(rnn_outputs).squeeze(-1), dim=1)
        return torch.bmm(attn_weights.unsqueeze(1), rnn_outputs).squeeze(1)


class GRUAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=1, bidirectional=True, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = Attention(gru_output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        return self.attention(outputs)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))

#LOAD VOCAB, LABELS, CHECKPOINT

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}
else:
    idx2label = {}

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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
print(f"Model loaded. Classes: {hp['num_classes']} | Vocab: {hp['vocab_size']}")


#PREDICTION

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def text_to_tensor(text: str) -> torch.Tensor:
    unk = word2idx["<UNK>"]
    ids = [word2idx.get(t, unk) for t in tokenize(text)]
    if len(ids) < MAX_LEN:
        ids = ids + [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return torch.tensor([ids], dtype=torch.long, device=device)


def label_for(class_id: int) -> str:
    return idx2label.get(class_id, f"Class {class_id}")


def predict_abstract(text: str):
    if not text or not text.strip():
        return {}, "Please paste an abstract above."

    tensor_input = text_to_tensor(text)
    with torch.no_grad():
        feats = encoder(tensor_input)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(TOP_K, probs.shape[0])
    top_conf, top_idx = torch.topk(probs, k=k)

    label_scores = {
        label_for(int(i.item())): float(c.item())
        for c, i in zip(top_conf, top_idx)
    }

    best_id = int(top_idx[0].item())
    best_conf = float(top_conf[0].item()) * 100
    summary = (
        f"**Predicted Class:** {label_for(best_id)} "
        f"(ID `{best_id}`) — **Confidence:** {best_conf:.2f}%"
    )
    return label_scores, summary



#INTERFACE

DESCRIPTION = (
    "Resource-efficient **Attention-GRU + GloVe-300d** model for fine-grained "
    "academic abstract classification, trained on the **WOS-46985** benchmark "
    "(134 sub-disciplines).\n\n"
    "Paste an abstract below to see the top-5 predicted disciplines."
)

EXAMPLES = [
    ["The exponential growth of scholarly literature necessitates automated systems "
     "for organizing scientific knowledge across disciplines."],
    ["We propose a novel attention-based recurrent architecture that leverages "
     "pretrained word embeddings to classify biomedical abstracts."],
    ["This study examines the macroeconomic effects of monetary policy shocks "
     "on emerging market economies using a structural VAR model."],
]

with gr.Blocks(title="Sustainable Science Mapping") as app:
    gr.Markdown("# Sustainable Science Mapping: Abstract Classifier")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                lines=8,
                placeholder="Paste an academic abstract here...",
                label="Abstract",
            )
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[inp])
        with gr.Column():
            summary_out = gr.Markdown(label="Result")
            topk_out = gr.Label(num_top_classes=TOP_K, label=f"Top-{TOP_K} Predictions")

    btn.click(predict_abstract, inputs=[inp], outputs=[topk_out, summary_out])
    inp.submit(predict_abstract, inputs=[inp], outputs=[topk_out, summary_out])

if __name__ == "__main__":
    app.launch()
