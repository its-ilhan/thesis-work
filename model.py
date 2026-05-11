import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BERT_DIM     = 768
NUMERIC_DIM  = 43
FEATURE_DIM  = BERT_DIM + NUMERIC_DIM  # 811

DROPOUT      = 0.4


# ═══════════════════════════════════════════════════════════════
# BRANCH 1: Acoustic MLP Branch
# Processes the 43 independent tabular features through a proper
# MLP — Conv1d was wrong here since there is no spatial/temporal
# relationship between independent stats like mel_mean, filler_count
# ═══════════════════════════════════════════════════════════════

class AcousticBranch(nn.Module):
    def __init__(self, input_dim: int = NUMERIC_DIM, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 43) → (batch, 128)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# BRANCH 2: Linguistic Branch
# Processes the BERT CLS embedding through a projection MLP
# LSTM was overkill for a single CLS vector — replaced with MLP
# ═══════════════════════════════════════════════════════════════

class LinguisticBranch(nn.Module):
    def __init__(self, input_dim: int = BERT_DIM, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 768) → (batch, 128)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# GATED MULTIMODAL FUSION
# Replaces the broken CrossAttention that was applying softmax
# over a sequence length of 1 (attention weights always = 1.0).
# This learns a soft gate: how much of each modality to trust
# for each sample, based on both representations together.
# ═══════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        # Gate takes both representations and learns how to weight them
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 2),   # 2 gate values — one per modality
            nn.Softmax(dim=-1)
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(
        self,
        linguistic: torch.Tensor,  # (batch, dim)
        acoustic:   torch.Tensor,  # (batch, dim)
    ) -> torch.Tensor:
        combined = torch.cat([linguistic, acoustic], dim=1)  # (batch, dim*2)
        gates    = self.gate(combined)                        # (batch, 2)

        g_ling = gates[:, 0:1]   # (batch, 1)
        g_acou = gates[:, 1:2]   # (batch, 1)

        fused = g_ling * linguistic + g_acou * acoustic       # (batch, dim)
        return self.drop(fused)


# ═══════════════════════════════════════════════════════════════
# FULL BI-MODAL MODEL
# ═══════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        bert_dim:    int = BERT_DIM,
        numeric_dim: int = NUMERIC_DIM,
        branch_dim:  int = 128,
    ):
        super().__init__()

        self.acoustic_branch   = AcousticBranch(numeric_dim, branch_dim)
        self.linguistic_branch = LinguisticBranch(bert_dim, branch_dim)
        self.fusion            = GatedFusion(branch_dim)

        # Classifier takes: linguistic + acoustic + fused
        fusion_input_dim = branch_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        bert_features:    torch.Tensor,  # (batch, 768)
        numeric_features: torch.Tensor,  # (batch, 43)
    ) -> torch.Tensor:

        acoustic   = self.acoustic_branch(numeric_features)    # (batch, 128)
        linguistic = self.linguistic_branch(bert_features)     # (batch, 128)
        fused      = self.fusion(linguistic, acoustic)         # (batch, 128)

        combined = torch.cat([linguistic, acoustic, fused], dim=1)  # (batch, 384)
        logits   = self.classifier(combined)                         # (batch, 1)
        return logits


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