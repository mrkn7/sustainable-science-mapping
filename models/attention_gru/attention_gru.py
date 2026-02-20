import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs):
        attn_weights = torch.softmax(self.attention(rnn_outputs).squeeze(-1), dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context_vector

class GRUAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=0.0, emb_matrix=None):
        super(GRUAttentionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if emb_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = Attention(gru_output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        context_vector = self.attention(outputs)
        return context_vector

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))