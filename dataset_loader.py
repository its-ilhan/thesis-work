import os
import whisper
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000
CHUNK_DURATION  = 10          # seconds per chunk
DATASET_ROOT    = "/content/for-norm"   # adjust if your FoR path differs on Colab
OUTPUT_DIR      = "/content/processed"
WHISPER_MODEL   = "base"               # use "small" if accuracy needs improvement

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: Audio Preprocessing
# ─────────────────────────────────────────────
def preprocess_audio(input_path: str, output_path: str) -> bool:
    """
    Load audio, convert to mono 16kHz, save as .wav
    Returns True on success, False on failure.
    """
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"  [ERROR] Could not process {input_path}: {e}")
        return False


def chunk_audio(audio_path: str):
    """
    Split a wav file into CHUNK_DURATION-second segments.
    Yields (chunk_array, start_time_seconds) tuples.
    """
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_size = CHUNK_DURATION * sr

    for i, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < sr:          # skip chunks shorter than 1 second
            continue
        yield chunk, round(start / sr, 3)


# ─────────────────────────────────────────────
# STEP 2: Transcription with Word Timestamps
# ─────────────────────────────────────────────
def transcribe_chunk(model, chunk: np.ndarray) -> dict:
    """
    Transcribe a single audio chunk using Whisper.
    Returns a dict with full text and word-level timestamps.
    """
    # Whisper expects float32 numpy array at 16kHz
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
                "start": round(word_info["start"], 3),   # seconds
                "end":   round(word_info["end"],   3),
            })

    return {
        "text":  result.get("text", "").strip(),
        "words": words
    }


# ─────────────────────────────────────────────
# STEP 3: Walk the FoR Dataset & Build Records
# ─────────────────────────────────────────────
def get_label_from_path(filepath: str) -> int:
    """
    FoR folder structure:
      for-norm/
        training/real/...
        training/fake/...
        testing/real/...
        testing/fake/...
    Returns 1 for real, 0 for fake.
    """
    parts = filepath.lower().replace("\\", "/").split("/")
    if "real" in parts:
        return 1
    elif "fake" in parts:
        return 0
    else:
        raise ValueError(f"Cannot determine label from path: {filepath}")


def build_dataset(whisper_model_size: str = WHISPER_MODEL) -> pd.DataFrame:
    """
    Main pipeline runner for Phase 1.
    Walks DATASET_ROOT, preprocesses audio, chunks it,
    transcribes each chunk, and saves a metadata CSV.
    """
    print(f"Loading Whisper model: {whisper_model_size}")
    model = whisper.load_model(whisper_model_size)

    records = []
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    # Collect all audio files
    all_files = []
    for root, _, files in os.walk(DATASET_ROOT):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in audio_extensions:
                all_files.append(os.path.join(root, fname))

    print(f"Found {len(all_files)} audio files. Starting Phase 1 pipeline...\n")

    for filepath in tqdm(all_files, desc="Processing files"):
        try:
            label = get_label_from_path(filepath)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        # ── Preprocess ──
        clean_path = os.path.join(
            OUTPUT_DIR,
            "clean_" + os.path.basename(filepath).replace(" ", "_")
        )
        clean_path = os.path.splitext(clean_path)[0] + ".wav"

        if not preprocess_audio(filepath, clean_path):
            continue

        # ── Chunk & Transcribe ──
        for chunk_idx, (chunk, start_time) in enumerate(chunk_audio(clean_path)):
            transcription = transcribe_chunk(model, chunk)

            chunk_filename = f"chunk_{chunk_idx}_{os.path.basename(clean_path)}"
            chunk_out_path = os.path.join(OUTPUT_DIR, chunk_filename)
            sf.write(chunk_out_path, chunk, SAMPLE_RATE)

            records.append({
                "original_file":  filepath,
                "chunk_file":     chunk_out_path,
                "label":          label,          # 1=real, 0=fake
                "chunk_index":    chunk_idx,
                "start_time_s":   start_time,
                "transcript":     transcription["text"],
                "word_timestamps": str(transcription["words"]),  # stored as string; parsed later
            })

    # ── Save metadata ──
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "phase1_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Phase 1 complete. {len(df)} chunks saved to: {csv_path}")
    return df


# ─────────────────────────────────────────────
# Entry point (for local testing only)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = build_dataset()
    print(df.head())