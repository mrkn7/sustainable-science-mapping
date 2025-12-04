import os
import torch
import torch.nn as nn
import re
import gradio as gr

# ---------------------------------------------------------
# 1. SETUP PATHS & DEVICE
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "gru_classifier.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ---------------------------------------------------------
# 2. MODEL DEFINITIONS (Must match training exactly)
# ---------------------------------------------------------
class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.directions = 2 if bidirectional else 1

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        if self.directions == 1:
            return h[-1]
        else:
            return torch.cat((h[-2], h[-1]), dim=1)

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# ---------------------------------------------------------
# 3. LOAD CHECKPOINT & INITIALIZE MODEL
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Please verify the 'models' folder structure.")

# Load checkpoint (Use map_location to ensure it works on CPU machines too)
checkpoint = torch.load(MODEL_PATH, map_location=device)

word2idx = checkpoint["word2idx"]
idx2label = checkpoint["idx2label"]

# Hyperparameters (Must match training config)
VOCAB_SIZE = len(word2idx)
EMBED_DIM = 100
HIDDEN_DIM = 64
BIDIR = True
input_dim = HIDDEN_DIM * (2 if BIDIR else 1)

# Initialize Models
encoder = GRUEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, bidirectional=BIDIR)
classifier = Classifier(input_dim)

# Load Weights
encoder.load_state_dict(checkpoint["encoder_state"])
classifier.load_state_dict(checkpoint["classifier_state"])

# Move to device and set to Eval mode
encoder.to(device)
classifier.to(device)
encoder.eval()
classifier.eval()

print("Model loaded successfully!")

# ---------------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------------
def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def text_to_ids(text, max_len=100):
    tokens = tokenize(text)
    ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]

    # Padding / Truncation
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def predict_abstract(text):
    if not text:
        return "Please enter an abstract."
        
    ids = text_to_ids(text).to(device)

    with torch.no_grad():
        features = encoder(ids)
        logits = classifier(features)
        
        # Get Probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
        label = idx2label[pred.item()]
        confidence = conf.item() * 100

    return f"Prediction: {label} (Confidence: {confidence:.2f}%)"

# ---------------------------------------------------------
# 5. GRADIO INTERFACE
# ---------------------------------------------------------
app = gr.Interface(
    fn=predict_abstract,
    inputs=gr.Textbox(lines=5, placeholder="Paste an academic abstract here...", label="Abstract"),
    outputs=gr.Textbox(label="Result"),
    title="Sustainable Science Mapping: Abstract Classifier 🌿",
    description="This interface uses a resource-efficient **GRU + GloVe** model to classify academic abstracts into AI, Economics, or Psychology.",
    examples=[
        ["Deep learning models have revolutionized natural language processing."],
        ["Market equilibrium is determined by supply and demand forces."],
        ["Cognitive dissonance theory explains the mental discomfort experienced by a person."]
    ],
    theme="default"
)

if __name__ == "__main__":
    app.launch()