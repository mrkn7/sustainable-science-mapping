import torch
import json
import re
from models.attention_gru import GRUAttentionEncoder, Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/attention_gru/word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)


checkpoint = torch.load("smodels/attention_gru/attention_gru_wos.pth", map_location=device)
hp = checkpoint['hyperparameters']

encoder = GRUAttentionEncoder(
    vocab_size=hp['vocab_size'],
    embed_dim=hp['embed_dim'],
    hidden_dim=hp['hidden_dim'],
    num_layers=hp['num_layers'],
    bidirectional=hp['bidirectional']
).to(device)

clf_input_dim = hp['hidden_dim'] * (2 if hp['bidirectional'] else 1)
classifier = Classifier(
    input_dim=clf_input_dim, 
    num_classes=hp['num_classes'], 
    dropout=hp['fc_dropout']
).to(device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

encoder.eval()
classifier.eval()

def predict_abstract(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    
    max_len = 250
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        
    tensor_input = torch.LongTensor([ids]).to(device)
    
    with torch.no_grad():
        feats = encoder(tensor_input)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
    return pred_class, confidence

