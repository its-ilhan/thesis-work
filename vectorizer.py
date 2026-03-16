import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR   = "/content/processed"
SCALER_PATH  = "/content/processed/scaler.pkl"
IMPUTER_PATH = "/content/processed/imputer.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# All numeric feature columns produced by Phase 2 and Phase 3
# BERT embeddings are handled separately
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
# STEP 1: Text Embedding with BERT
# ─────────────────────────────────────────────

def load_bert():
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_bert_embedding(text: str, tokenizer, model) -> np.ndarray:
    """
    Passes transcript text through BERT and returns the [CLS] token
    embedding as a 768-dimensional vector. The CLS token is BERT's
    summary representation of the entire input sentence.
    Returns zeros if text is empty.
    """
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

    # CLS token is the first token of the last hidden state
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 2: Numeric Feature Cleaning & Scaling
# ─────────────────────────────────────────────

def clean_numeric_features(df: pd.DataFrame, fit: bool = True) -> np.ndarray:
    """
    Extracts all numeric feature columns, imputes any missing values
    with column means, then standard-scales everything to zero mean
    and unit variance.

    fit=True  → fit the imputer and scaler on this data (training set)
    fit=False → use already-fitted imputer and scaler (validation/test)
    """
    # Keep only the feature columns that exist in this dataframe
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(np.float64)

    if fit:
        imputer = SimpleImputer(strategy="mean")
        scaler  = StandardScaler()
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)
        joblib.dump(imputer, IMPUTER_PATH)
        joblib.dump(scaler,  SCALER_PATH)
        print(f"  Imputer and scaler saved to {OUTPUT_DIR}")
    else:
        imputer = joblib.load(IMPUTER_PATH)
        scaler  = joblib.load(SCALER_PATH)
        X = imputer.transform(X)
        X = scaler.transform(X)

    return X.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 3: Feature Concatenation
# ─────────────────────────────────────────────

def build_feature_vector(
    bert_embedding: np.ndarray,
    numeric_features: np.ndarray
) -> np.ndarray:
    """
    Concatenates the BERT embedding (768-dim) with the scaled numeric
    features (n-dim) into a single 1D feature vector per chunk.
    This is the final input vector that gets fed into the model.
    """
    return np.concatenate([bert_embedding, numeric_features]).astype(np.float32)


# ─────────────────────────────────────────────
# MASTER FUNCTION: Run All of Phase 4
# ─────────────────────────────────────────────

def build_phase4_vectors(
    phase3_csv: str  = "/content/processed/phase3_features.csv",
    output_path: str = "/content/processed/phase4_vectors.npz",
    fit_scalers: bool = True
) -> dict:
    """
    Reads Phase 3 CSV, builds BERT embeddings for every transcript,
    cleans and scales all numeric features, concatenates them into
    one final feature vector per chunk, and saves everything as a
    .npz file (compressed numpy arrays).

    Returns a dict with keys: X (feature matrix), y (labels),
    feature_dim (total vector size), chunk_files (for reference).
    """
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

    bert_matrix = np.stack(bert_embeddings, axis=0)  # shape: (N, 768)
    print(f"  BERT matrix shape: {bert_matrix.shape}")

    # ── Numeric features ──
    print("\nScaling numeric features...")
    numeric_matrix = clean_numeric_features(df, fit=fit_scalers)
    print(f"  Numeric matrix shape: {numeric_matrix.shape}")

    # ── Concatenate ──
    print("\nConcatenating feature vectors...")
    X = np.concatenate([bert_matrix, numeric_matrix], axis=1)
    y = df["label"].values.astype(np.int64)

    print(f"  Final feature matrix shape: {X.shape}")
    print(f"  Labels — Real: {np.sum(y==1)}, Fake: {np.sum(y==0)}")

    # ── Save as compressed numpy file ──
    chunk_files = df["chunk_file"].values
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        chunk_files=chunk_files
    )
    print(f"\n✅ Phase 4 complete. Vectors saved to: {output_path}")

    return {
        "X":           X,
        "y":           y,
        "feature_dim": X.shape[1],
        "chunk_files": chunk_files
    }


if __name__ == "__main__":
    result = build_phase4_vectors()
    print(f"Feature vector dimension: {result['feature_dim']}")