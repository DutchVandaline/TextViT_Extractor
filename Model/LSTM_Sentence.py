import torch.nn as nn
import torch


class SentenceLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.3, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, num_classes)
        )
    
    def forward(self, x, attention_mask=None):
        # x: (batch, num_sentences, seq_len)
        batch_size, num_sentences, seq_len = x.shape
        x = x.view(batch_size * num_sentences, seq_len)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size * num_sentences, seq_len)
        

        embeds = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        
        sentence_repr = lstm_out.mean(dim=1)  # (B*num_sentences, hidden_dim*directions)
        sentence_repr = self.dropout(sentence_repr)
        
        logits = self.classifier(sentence_repr)  # (B*num_sentences, 1)
        logits = logits.view(batch_size, num_sentences)
        return logits