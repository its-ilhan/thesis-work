import os
import ast
import whisper
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG — matches your Colab folder structure
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000
CHUNK_DURATION  = 10
DATASET_ROOT    = "/content/thesis-work/dataset/for-norm"
OUTPUT_DIR      = "/content/processed"
WHISPER_MODEL   = "base"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: Audio Preprocessing
# ─────────────────────────────────────────────
def preprocess_audio(input_path: str, output_path: str) -> bool:
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"  [ERROR] Could not process {input_path}: {e}")
        return False


def chunk_audio(audio_path: str):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_size = CHUNK_DURATION * sr
    for i, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < sr:
            continue
        yield chunk, round(start / sr, 3)


# ─────────────────────────────────────────────
# STEP 2: Transcription with Word Timestamps
# ─────────────────────────────────────────────
def transcribe_chunk(model, chunk: np.ndarray) -> dict:
    result = model.transcribe(
        chunk.astype(np.float32),
        word_timestamps=True,
        language="en"
    )
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word":  word_info["word"].strip(),
                "start": round(word_info["start"], 3),
                "end":   round(word_info["end"],   3),
            })
    return {
        "text":  result.get("text", "").strip(),
        "words": words
    }


# ─────────────────────────────────────────────
# STEP 3: Label from path
# ─────────────────────────────────────────────
def get_label_from_path(filepath: str) -> int:
    """
    FoR folder structure:
      for-norm/
        training/real/   → label 1
        training/fake/   → label 0
        testing/real/    → label 1
        testing/fake/    → label 0
    """
    parts = filepath.lower().replace("\\", "/").split("/")
    if "real" in parts:
        return 1
    elif "fake" in parts:
        return 0
    else:
        raise ValueError(f"Cannot determine label from path: {filepath}")


# ─────────────────────────────────────────────
# MAIN: Build Dataset
# ─────────────────────────────────────────────
def build_dataset(whisper_model_size: str = WHISPER_MODEL) -> pd.DataFrame:
    print(f"Loading Whisper '{whisper_model_size}' model...")
    model = whisper.load_model(whisper_model_size)

    records = []
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    all_files = []
    for root, _, files in os.walk(DATASET_ROOT):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in audio_extensions:
                all_files.append(os.path.join(root, fname))

    print(f"Found {len(all_files)} audio files.\n")

    # ── Safety cap for testing: remove or increase this later ──
    all_files = all_files[:20]
    print("⚠️  Running on first 20 files only for testing. Remove cap when ready.\n")

    for filepath in tqdm(all_files, desc="Phase 1 Progress"):
        try:
            label = get_label_from_path(filepath)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        clean_name = "clean_" + os.path.basename(filepath).replace(" ", "_")
        clean_name = os.path.splitext(clean_name)[0] + ".wav"
        clean_path = os.path.join(OUTPUT_DIR, clean_name)

        if not preprocess_audio(filepath, clean_path):
            continue

        for chunk_idx, (chunk, start_time) in enumerate(chunk_audio(clean_path)):
            transcription = transcribe_chunk(model, chunk)

            chunk_filename = f"chunk{chunk_idx}_{clean_name}"
            chunk_out_path = os.path.join(OUTPUT_DIR, chunk_filename)
            sf.write(chunk_out_path, chunk, SAMPLE_RATE)

            records.append({
                "original_file":   filepath,
                "chunk_file":      chunk_out_path,
                "label":           label,
                "chunk_index":     chunk_idx,
                "start_time_s":    start_time,
                "transcript":      transcription["text"],
                "word_timestamps": str(transcription["words"]),
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "phase1_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Phase 1 done! {len(df)} chunks saved to: {csv_path}")
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df[["label", "transcript", "chunk_file"]].head())