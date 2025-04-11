import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072,
                 mlp_dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim,
                                                     num_heads,
                                                     attn_dropout)
        self.mlp_block = MultiLayerPerceptronLayer(embedding_dim,
                                                   mlp_size,
                                                   mlp_dropout)

    def forward(self, x, attention_mask=None):
        x = self.msa_block(x, attention_mask) + x
        x = self.mlp_block(x) + x
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x, attention_mask=None):
        x_norm = self.layer_norm(x)
        # attention_mask: (batch, seq_len), 1 for real tokens, 0 for pad
        key_padding_mask = None
        if attention_mask is not None:
            # invert mask: True where we want to ignore
            key_padding_mask = attention_mask == 0
        attn_output, _ = self.multihead_attn(query=x_norm,
                                             key=x_norm,
                                             value=x_norm,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=False)
        return attn_output

class MultiLayerPerceptronLayer(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.mlp(x_norm)

class SentenceExtractor(nn.Module):
    def __init__(self, vocab_size, max_seq_len,
                 num_transformer_layers=4,
                 embedding_dim=768, mlp_size=3072,
                 num_heads=12, attn_dropout=0.1,
                 mlp_dropout=0.1, embedding_dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim,
                                    num_heads,
                                    mlp_size,
                                    mlp_dropout,
                                    attn_dropout)
            for _ in range(num_transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x, attention_mask=None):
        # x: (batch, num_sentences, seq_len)
        batch_size, num_sentences, seq_len = x.shape
        x = x.view(batch_size * num_sentences, seq_len)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size * num_sentences, seq_len)

        token_embeds = self.token_embedding(x)
        pos_embeds = self.position_embedding[:, :seq_len, :]
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        cls_tokens = x[:, 0, :]                      # (B*S, D)
        logits = self.classifier(cls_tokens)         # (B*S, 1)
        logits = logits.view(batch_size, num_sentences)
        return logits
