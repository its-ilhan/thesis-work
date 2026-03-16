import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BERT_DIM      = 768   # BERT CLS embedding size
NUMERIC_DIM   = 43    # Phase 2 + Phase 3 numeric features
FEATURE_DIM   = BERT_DIM + NUMERIC_DIM  # 811 — full input vector size

CNN_CHANNELS  = 64    # number of filters in the audio CNN branch
LSTM_HIDDEN   = 128   # hidden size of the linguistic LSTM
ATTN_HEADS    = 4     # number of attention heads in cross-attention
DROPOUT       = 0.3


# ═══════════════════════════════════════════════════════════════
# BRANCH 1: Audio CNN Branch
# Processes the Mel/acoustic numeric features to find low-level
# synthetic artifacts in the sound
# ═══════════════════════════════════════════════════════════════

class AudioBranch(nn.Module):
    def __init__(self, input_dim: int = NUMERIC_DIM, output_dim: int = 128):
        super().__init__()

        # Treat the 43 numeric features as a 1D sequence of length 43
        # Conv1d expects (batch, channels, length)
        self.conv1 = nn.Conv1d(1, CNN_CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS * 2, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.norm1 = nn.BatchNorm1d(CNN_CHANNELS)
        self.norm2 = nn.BatchNorm1d(CNN_CHANNELS * 2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(DROPOUT)
        self.fc    = nn.Linear(CNN_CHANNELS * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, numeric_dim)
        x = x.unsqueeze(1)                      # (batch, 1, numeric_dim)
        x = self.relu(self.norm1(self.conv1(x))) # (batch, 64, numeric_dim)
        x = self.relu(self.norm2(self.conv2(x))) # (batch, 128, numeric_dim)
        x = self.pool(x).squeeze(-1)             # (batch, 128)
        x = self.drop(x)
        x = self.fc(x)                           # (batch, output_dim)
        return x


# ═══════════════════════════════════════════════════════════════
# BRANCH 2: Linguistic Branch
# Processes the BERT embeddings + linguistic features through
# an LSTM to capture language flow patterns
# ═══════════════════════════════════════════════════════════════

class LinguisticBranch(nn.Module):
    def __init__(self, input_dim: int = BERT_DIM, output_dim: int = 128):
        super().__init__()

        self.proj = nn.Linear(input_dim, 256)   # project BERT down first
        self.norm = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)

        # Bidirectional LSTM — reads the projected embedding both forward
        # and backward to capture full context
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )

        # BiLSTM output is LSTM_HIDDEN * 2 because of bidirectionality
        self.fc = nn.Linear(LSTM_HIDDEN * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, bert_dim)
        x = self.relu(self.norm(self.proj(x)))  # (batch, 256)
        x = self.drop(x)
        x = x.unsqueeze(1)                      # (batch, 1, 256) — seq length 1
        lstm_out, _ = self.lstm(x)              # (batch, 1, 256)
        x = lstm_out[:, -1, :]                  # take last timestep (batch, 256)
        x = self.fc(x)                          # (batch, output_dim)
        return x


# ═══════════════════════════════════════════════════════════════
# CROSS-ATTENTION MODULE
# Forces the model to look at acoustic features through the lens
# of the linguistic features — "does this sound match these words?"
# ═══════════════════════════════════════════════════════════════

class CrossAttention(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        # Query comes from linguistic branch (text)
        # Key and Value come from audio branch (sound)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj   = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj   = nn.Linear(dim, dim)
        self.scale      = dim ** 0.5
        self.drop       = nn.Dropout(DROPOUT)

    def forward(
        self,
        linguistic: torch.Tensor,  # (batch, dim) — the query
        acoustic:   torch.Tensor,  # (batch, dim) — key and value
    ) -> torch.Tensor:

        Q = self.query_proj(linguistic).unsqueeze(1)  # (batch, 1, dim)
        K = self.key_proj(acoustic).unsqueeze(1)      # (batch, 1, dim)
        V = self.value_proj(acoustic).unsqueeze(1)    # (batch, 1, dim)

        # Scaled dot-product attention
        attn_scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, 1, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        attended = torch.bmm(attn_weights, V).squeeze(1)  # (batch, dim)
        out      = self.out_proj(attended)                 # (batch, dim)
        return out


# ═══════════════════════════════════════════════════════════════
# FULL BI-MODAL MODEL
# Combines both branches + cross-attention + classification head
# ═══════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        bert_dim:    int = BERT_DIM,
        numeric_dim: int = NUMERIC_DIM,
        branch_dim:  int = 128,
    ):
        super().__init__()

        self.audio_branch      = AudioBranch(numeric_dim, branch_dim)
        self.linguistic_branch = LinguisticBranch(bert_dim, branch_dim)
        self.cross_attention   = CrossAttention(branch_dim)

        # After cross-attention we have three representations to classify from:
        # linguistic output + acoustic output + cross-attended output
        fusion_dim = branch_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)   # binary output — real vs fake
        )

    def forward(
        self,
        bert_features:    torch.Tensor,  # (batch, 768)
        numeric_features: torch.Tensor,  # (batch, 43)
    ) -> torch.Tensor:

        acoustic   = self.audio_branch(numeric_features)      # (batch, 128)
        linguistic = self.linguistic_branch(bert_features)    # (batch, 128)
        cross      = self.cross_attention(linguistic, acoustic) # (batch, 128)

        # Concatenate all three representations
        fused  = torch.cat([linguistic, acoustic, cross], dim=1)  # (batch, 384)
        logits = self.classifier(fused)                            # (batch, 1)
        return logits


# ─────────────────────────────────────────────
# HELPER: Build model and print summary
# ─────────────────────────────────────────────

def build_model(device: str = "cpu") -> DeepfakeDetector:
    model = DeepfakeDetector(
        bert_dim=BERT_DIM,
        numeric_dim=NUMERIC_DIM,
        branch_dim=128
    )
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model built successfully.")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device              : {device}")
    return model


if __name__ == "__main__":
    model = build_model()
    print(model)