import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = "/content/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    # Phase 2 — Lexical
    "filler_count", "filler_rate", "repeated_word_count", "false_start_count",
    "unique_word_ratio", "avg_word_length", "sentence_count", "avg_sentence_length",
    # Phase 2 — Prosodic
    "pause_count", "avg_pause_duration", "max_pause_duration",
    "unnatural_pause_count", "missing_pause_after_long",
    "pause_at_conjunction", "pause_variance",
    # Phase 2 — Sentiment
    "sentiment_score", "sentiment_value",
    # Phase 3 — Mel
    "mel_mean", "mel_std", "mel_min", "mel_max", "mel_bandwidth", "mel_flatness",
    # Phase 3 — Pitch
    "pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
    "pitch_slope_mean", "pitch_slope_std", "voiced_fraction", "pitch_on_filler",
    # Phase 3 — Energy
    "energy_mean", "energy_std", "energy_min", "energy_max", "energy_range",
    "energy_slope_mean", "energy_slope_std", "energy_flatness",
    # Phase 3 — Mismatch
    "emotion_intensity", "acoustic_flatness", "mismatch_score",
]


# ─────────────────────────────────────────────
# BERT Embedding
# ─────────────────────────────────────────────

def load_bert():
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_bert_embedding(text: str, tokenizer, model) -> np.ndarray:
    import torch
    if not text or str(text).strip() == "":
        return np.zeros(768, dtype=np.float32)

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.astype(np.float32)


# ─────────────────────────────────────────────
# MASTER FUNCTION
# Saves raw (unscaled) BERT + numeric matrices.
# Scaling happens inside train() AFTER the split
# to prevent data leakage.
# ─────────────────────────────────────────────

def build_phase4_vectors(
    phase3_csv:  str = "/content/processed/phase3_features.csv",
    output_path: str = "/content/processed/phase4_vectors.npz",
) -> dict:

    print("Loading Phase 3 data...")
    df = pd.read_csv(phase3_csv)
    print(f"  {len(df)} chunks loaded.\n")

    # ── BERT embeddings ──
    print("Loading BERT model...")
    tokenizer, bert_model = load_bert()

    bert_embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BERT Embedding"):
        transcript = str(row.get("transcript", ""))
        emb = get_bert_embedding(transcript, tokenizer, bert_model)
        bert_embeddings.append(emb)

    bert_matrix = np.stack(bert_embeddings, axis=0)
    print(f"  BERT matrix shape: {bert_matrix.shape}")

    # ── Raw numeric features — NO scaling here ──
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing feature cols: {missing}")

    numeric_matrix = df[available].copy().values.astype(np.float32)
    print(f"  Numeric matrix shape (raw): {numeric_matrix.shape}")

    y           = df["label"].values.astype(np.int64)
    chunk_files = df["chunk_file"].values

    print(f"\n  Labels — Real: {np.sum(y==1)}, Fake: {np.sum(y==0)}")

    np.savez_compressed(
        output_path,
        bert=bert_matrix,
        numeric=numeric_matrix,
        y=y,
        chunk_files=chunk_files
    )

    print(f"\n✅ Phase 4 complete. Saved to: {output_path}")
    print(f"   BERT dim: {bert_matrix.shape[1]}, Numeric dim: {numeric_matrix.shape[1]}")

    return {
        "bert":        bert_matrix,
        "numeric":     numeric_matrix,
        "y":           y,
        "chunk_files": chunk_files,
    }


if __name__ == "__main__":
    build_phase4_vectors()