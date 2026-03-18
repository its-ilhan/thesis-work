# ═══════════════════════════════════════════════════════════════
# dataset_loader.py
# Covers Phase 1 (audio prep + transcription) and
#         Phase 2 (NLP linguistic feature extraction)
# ═══════════════════════════════════════════════════════════════

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import soundfile as sf
import whisper

import spacy
from transformers import pipeline

# ── Load heavy models once at module level ──
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000
CHUNK_DURATION  = 10
DATASET_ROOT    = "/content/thesis-work/dataset"
OUTPUT_DIR      = "/content/processed"
WHISPER_MODEL   = "base"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: DATA PREPARATION & TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════

def preprocess_audio(input_path: str, output_path: str) -> bool:
    """Convert any audio file to 16kHz mono WAV."""
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"  [ERROR] Could not process {input_path}: {e}")
        return False


def chunk_audio(audio_path: str):
    """Split audio into CHUNK_DURATION-second segments."""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_size = CHUNK_DURATION * sr
    for i, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < sr:
            continue
        yield chunk, round(start / sr, 3)


def transcribe_chunk(model, chunk: np.ndarray) -> dict:
    """Transcribe audio chunk with Whisper, returning text + word timestamps."""
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


def get_label_from_path(filepath: str) -> int:
    """Return 1 for real, 0 for fake based on folder name."""
    parts = filepath.lower().replace("\\", "/").split("/")
    if "real" in parts:
        return 1
    elif "fake" in parts:
        return 0
    else:
        raise ValueError(f"Cannot determine label from path: {filepath}")


def build_dataset(whisper_model_size: str = WHISPER_MODEL) -> pd.DataFrame:
    """Master Phase 1 function — preprocesses, chunks, transcribes all audio."""
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

    # ── Safety cap: remove when ready for full dataset ──
    # Separate real and fake files
    real_files = [f for f in all_files if "real" in f.lower().replace("\\", "/").split("/")]
    fake_files = [f for f in all_files if "fake" in f.lower().replace("\\", "/").split("/")]
    

    #Run full dataset
    # all_files = real_files + fake_files
    # print(f"Full dataset: {len(real_files)} real + {len(fake_files)} fake files.\n")

    CAP = 9000  # adjust this number based on available time

    real_files = real_files[5000:CAP]
    fake_files = fake_files[5000:CAP]
    all_files  = real_files + fake_files

    print(f"Training run: {len(real_files)} real + {len(fake_files)} fake files.")
    print(f"Total files  : {len(all_files)}\n")



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


# ═══════════════════════════════════════════════════════════════
# PHASE 2: NLP & LINGUISTIC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "so", "basically",
    "literally", "actually", "right", "okay", "hmm", "ah", "er"
}

# ─────────────────────────────────────────────
# VECTOR 1: Lexical & Syntactic Anomalies
# ─────────────────────────────────────────────

def extract_lexical_features(transcript: str) -> dict:
    """
    Detects human speech 'messiness' — filler words, repeated words,
    false starts, vocabulary diversity. AI speech is unnaturally clean.
    """
    empty = {
        "filler_count": 0, "filler_rate": 0.0,
        "repeated_word_count": 0, "false_start_count": 0,
        "unique_word_ratio": 0.0, "avg_word_length": 0.0,
        "sentence_count": 0, "avg_sentence_length": 0.0
    }

    if not transcript or transcript.strip() == "":
        return empty

    doc         = nlp(transcript.lower())
    tokens      = [t for t in doc if not t.is_space]
    words       = [t.text for t in tokens]
    total_words = len(words)

    if total_words == 0:
        return empty

    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_rate  = filler_count / total_words

    repeated_word_count = sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1]
    )

    false_start_count = 0
    for i, token in enumerate(tokens):
        if len(token.text) <= 3 and (i + 1) < len(tokens):
            if tokens[i + 1].text == ",":
                false_start_count += 1

    unique_word_ratio = len(set(words)) / total_words
    avg_word_length   = float(np.mean([len(w) for w in words]))

    sentences          = list(doc.sents)
    sentence_count     = len(sentences)
    avg_sentence_length = float(np.mean(
        [len([t for t in s if not t.is_space]) for s in sentences]
    )) if sentences else 0.0

    return {
        "filler_count":         filler_count,
        "filler_rate":          round(filler_rate, 4),
        "repeated_word_count":  repeated_word_count,
        "false_start_count":    false_start_count,
        "unique_word_ratio":    round(unique_word_ratio, 4),
        "avg_word_length":      round(avg_word_length, 4),
        "sentence_count":       sentence_count,
        "avg_sentence_length":  round(avg_sentence_length, 4),
    }


# ─────────────────────────────────────────────
# VECTOR 2: Prosodic-Linguistic Mapping
# ─────────────────────────────────────────────

def parse_word_timestamps(raw: str) -> list:
    """Parse the word_timestamps string from Phase 1 CSV into a list of dicts."""
    if not raw or str(raw).strip() in ("", "[]", "nan"):
        return []
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return []


def get_dependency_relation(word: str, doc) -> str:
    """Return spaCy dependency label for a word in the doc."""
    word_clean = word.strip().lower()
    for token in doc:
        if token.text.lower() == word_clean:
            return token.dep_
    return "unknown"


def extract_prosodic_features(transcript: str, word_timestamps: list) -> dict:
    """
    Calculates pause durations and checks whether pauses fall in
    natural positions (conjunctions, sentence ends) or unnatural ones
    (inside tight word pairs like adjective+noun). AI speech places
    pauses incorrectly because it has no breath or fatigue constraints.
    """
    empty = {
        "pause_count": 0, "avg_pause_duration": 0.0,
        "max_pause_duration": 0.0, "unnatural_pause_count": 0,
        "missing_pause_after_long": 0, "pause_at_conjunction": 0,
        "pause_variance": 0.0
    }

    if not word_timestamps or len(word_timestamps) < 2:
        return empty

    doc = nlp(transcript.lower()) if transcript else None

    TIGHT_DEPS   = {"amod", "det", "poss", "compound", "nummod"}
    CONJUNCTIONS = {
        "and", "but", "or", "because", "although",
        "however", "so", "yet", "for", "nor"
    }
    PAUSE_THRESHOLD = 0.15  # seconds

    pauses                   = []
    unnatural_pause_count    = 0
    pause_at_conjunction     = 0
    missing_pause_after_long = 0

    for i in range(1, len(word_timestamps)):
        prev_word = word_timestamps[i - 1]
        curr_word = word_timestamps[i]
        gap       = curr_word["start"] - prev_word["end"]

        if gap < PAUSE_THRESHOLD:
            continue

        pauses.append(gap)

        if doc:
            dep = get_dependency_relation(prev_word["word"], doc)
            if dep in TIGHT_DEPS:
                unnatural_pause_count += 1

        if curr_word["word"].strip().lower() in CONJUNCTIONS:
            pause_at_conjunction += 1

    if doc:
        for sent in doc.sents:
            sent_words = [t.text for t in sent if not t.is_space]
            if len(sent_words) > 15:
                last_word = sent_words[-1].lower()
                for j, wt in enumerate(word_timestamps[:-1]):
                    if wt["word"].strip().lower() == last_word:
                        next_gap = word_timestamps[j + 1]["start"] - wt["end"]
                        if next_gap < PAUSE_THRESHOLD:
                            missing_pause_after_long = 1
                        break

    if not pauses:
        return empty

    return {
        "pause_count":               len(pauses),
        "avg_pause_duration":        round(float(np.mean(pauses)), 4),
        "max_pause_duration":        round(float(np.max(pauses)), 4),
        "unnatural_pause_count":     unnatural_pause_count,
        "missing_pause_after_long":  missing_pause_after_long,
        "pause_at_conjunction":      pause_at_conjunction,
        "pause_variance":            round(float(np.var(pauses)), 4),
    }


# ─────────────────────────────────────────────
# VECTOR 3: Sentiment / Emotion Detection
# ─────────────────────────────────────────────

LABEL_TO_SCORE = {
    "positive": 1.0,  "neutral": 0.0,  "negative": -1.0,
    "label_0":  -1.0, "label_1": 0.0,  "label_2":   1.0,
}

def extract_sentiment_features(transcript: str) -> dict:
    """
    Runs RoBERTa sentiment on the transcript. This score is later
    cross-referenced against pitch/energy in Phase 3 — emotional text
    with flat acoustics is a strong deepfake signal.
    """
    empty = {
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "sentiment_value": 0.0
    }

    if not transcript or transcript.strip() == "":
        return empty

    try:
        result = sentiment_analyzer(transcript[:512])[0]
        label  = result["label"].lower()
        score  = round(result["score"], 4)
        value  = LABEL_TO_SCORE.get(label, 0.0)
        return {
            "sentiment_label": label,
            "sentiment_score": score,
            "sentiment_value": value,
        }
    except Exception as e:
        print(f"  [WARN] Sentiment failed: {e}")
        return empty


# ─────────────────────────────────────────────
# MASTER FUNCTION: Run All of Phase 2
# ─────────────────────────────────────────────

def build_phase2_features(
    phase1_csv: str = "/content/processed/phase1_metadata.csv",
    output_csv: str = "/content/processed/phase2_features.csv"
) -> pd.DataFrame:
    """
    Reads Phase 1 CSV, runs all three linguistic vectors on every row,
    saves enriched CSV with all new feature columns appended.
    """
    print("Loading Phase 1 data...")
    df = pd.read_csv(phase1_csv)
    print(f"  {len(df)} chunks loaded.\n")

    lexical_records   = []
    prosodic_records  = []
    sentiment_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phase 2 Progress"):
        transcript      = str(row.get("transcript", ""))
        word_timestamps = parse_word_timestamps(row.get("word_timestamps", "[]"))

        lexical_records.append(extract_lexical_features(transcript))
        prosodic_records.append(extract_prosodic_features(transcript, word_timestamps))
        sentiment_records.append(extract_sentiment_features(transcript))

    df_lexical   = pd.DataFrame(lexical_records)
    df_prosodic  = pd.DataFrame(prosodic_records)
    df_sentiment = pd.DataFrame(sentiment_records)

    df_out = pd.concat([df, df_lexical, df_prosodic, df_sentiment], axis=1)
    df_out.to_csv(output_csv, index=False)

    print(f"\n✅ Phase 2 complete. Saved to: {output_csv}")
    print(f"   Total columns now: {len(df_out.columns)}")
    return df_out


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())