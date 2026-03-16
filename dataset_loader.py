import os
import ast
import whisper
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import pipeline

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


# ═══════════════════════════════════════════════════════════════
# PHASE 2: NLP & LINGUISTIC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════


# ── Load models once at module level so they aren't reloaded per row ──
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True
)

# ─────────────────────────────────────────────
# VECTOR 1: Lexical & Syntactic Anomalies
# "The Perfect Speech Detector"
# ─────────────────────────────────────────────

FILLER_WORDS = {"um", "uh", "like", "you know", "i mean", "so", "basically",
                "literally", "actually", "right", "okay", "hmm", "ah", "er"}

def extract_lexical_features(transcript: str) -> dict:
    """
    Analyzes the transcript text for signs of human 'messiness'.
    
    Human speech is imperfect — it contains filler words, repeated words,
    and false starts. AI-generated speech is unnaturally clean because
    TTS models are trained to eliminate these. Their absence is a red flag.
    
    Returns:
        filler_count        : total number of filler words found
        filler_rate         : fillers per word (normalized for sentence length)
        repeated_word_count : how many times a word was immediately repeated
                              e.g. "I I went" counts as 1
        false_start_count   : short words (<=3 chars) followed by a comma,
                              a common pattern of self-correction in humans
        unique_word_ratio   : vocabulary diversity — humans vary their words
                              more naturally than some TTS systems
        avg_word_length     : stylistic feature; AI text tends to be more formal
        sentence_count      : number of sentences detected by spaCy
        avg_sentence_length : avg words per sentence
    """
    if not transcript or transcript.strip() == "":
        return {
            "filler_count": 0, "filler_rate": 0.0,
            "repeated_word_count": 0, "false_start_count": 0,
            "unique_word_ratio": 0.0, "avg_word_length": 0.0,
            "sentence_count": 0, "avg_sentence_length": 0.0
        }

    doc = nlp(transcript.lower())
    tokens = [t for t in doc if not t.is_space]
    words  = [t.text for t in tokens]
    total_words = len(words)

    if total_words == 0:
        return {
            "filler_count": 0, "filler_rate": 0.0,
            "repeated_word_count": 0, "false_start_count": 0,
            "unique_word_ratio": 0.0, "avg_word_length": 0.0,
            "sentence_count": 0, "avg_sentence_length": 0.0
        }

    # Filler word count
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_rate  = filler_count / total_words

    # Repeated consecutive words — e.g. "I I went to"
    repeated_word_count = sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1]
    )

    # False starts — short word immediately before a comma
    # e.g. "I, went to the store" → likely a self-correction
    false_start_count = sum(
        1 for token in tokens
        if len(token.text) <= 3 and token.nbor(1).text == ","
        if token.i + 1 < len(doc)
    )

    # Vocabulary diversity
    unique_word_ratio = len(set(words)) / total_words

    # Average word length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0

    # Sentence-level stats using spaCy sentence segmentation
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    avg_sentence_length = (
        np.mean([len([t for t in s if not t.is_space]) for s in sentences])
        if sentences else 0.0
    )

    return {
        "filler_count":        filler_count,
        "filler_rate":         round(filler_rate, 4),
        "repeated_word_count": repeated_word_count,
        "false_start_count":   false_start_count,
        "unique_word_ratio":   round(unique_word_ratio, 4),
        "avg_word_length":     round(float(avg_word_length), 4),
        "sentence_count":      sentence_count,
        "avg_sentence_length": round(float(avg_sentence_length), 4),
    }


# ─────────────────────────────────────────────
# VECTOR 2: Prosodic-Linguistic Mapping
# "The Breathing Detector"
# ─────────────────────────────────────────────

def parse_word_timestamps(raw: str) -> list:
    """
    Safely parses the word_timestamps string from the Phase 1 CSV
    back into a Python list of dicts.
    """
    if not raw or str(raw).strip() in ("", "[]", "nan"):
        return []
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return []


def get_dependency_relation(word: str, doc) -> str:
    """
    Given a word string and a spaCy doc, returns the dependency
    relation label of the first matching token (e.g. 'amod', 'nsubj').
    Returns 'unknown' if not found.
    """
    word_clean = word.strip().lower()
    for token in doc:
        if token.text.lower() == word_clean:
            return token.dep_
    return "unknown"


def extract_prosodic_features(transcript: str, word_timestamps: list) -> dict:
    """
    Maps silences between words against sentence structure to detect
    unnatural pause placement — a key deepfake indicator.

    Human speech rules:
      - Pauses happen at sentence boundaries, commas, and conjunctions
      - Pauses do NOT happen between tightly coupled word pairs like
        adjective+noun ("red car") or determiner+noun ("the book")
      - After a long sentence, humans MUST pause to breathe

    AI-generated speech violates these rules in detectable ways.

    Returns:
        pause_count              : total number of pauses detected
        avg_pause_duration       : mean silence duration in seconds
        max_pause_duration       : longest single silence
        unnatural_pause_count    : pauses inside tightly coupled word pairs
                                   (adjective→noun, determiner→noun, etc.)
        missing_pause_after_long : 1 if a long sentence (>15 words) has no
                                   pause at its end, 0 otherwise
        pause_at_conjunction     : count of natural pauses before conjunctions
                                   (and, but, or, because) — humans do this
        pause_variance           : variance in pause lengths; humans are
                                   inconsistent, AI tends to be uniform
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

    TIGHT_DEPS = {"amod", "det", "poss", "compound", "nummod"}
    CONJUNCTIONS = {"and", "but", "or", "because", "although", "however",
                    "so", "yet", "for", "nor"}
    PAUSE_THRESHOLD = 0.15  # seconds — gaps below this are not real pauses

    pauses = []
    unnatural_pause_count    = 0
    pause_at_conjunction     = 0
    missing_pause_after_long = 0

    for i in range(1, len(word_timestamps)):
        prev_word = word_timestamps[i - 1]
        curr_word = word_timestamps[i]

        gap = curr_word["start"] - prev_word["end"]

        if gap < PAUSE_THRESHOLD:
            continue  # not a real pause, just natural word flow

        pauses.append(gap)

        # ── Check if this pause is in a tightly coupled position ──
        # e.g. pause between "beautiful" (amod) and "sunset" (noun)
        # That would be unnatural for both humans AND AI, but AI does it
        # because it processes tokens independently without breath awareness
        if doc:
            dep = get_dependency_relation(prev_word["word"], doc)
            if dep in TIGHT_DEPS:
                unnatural_pause_count += 1

        # ── Check if the pause is naturally before a conjunction ──
        curr_text = curr_word["word"].strip().lower()
        if curr_text in CONJUNCTIONS:
            pause_at_conjunction += 1

    # ── Check for missing pause after a long sentence ──
    if doc:
        for sent in doc.sents:
            sent_words = [t.text for t in sent if not t.is_space]
            if len(sent_words) > 15:
                # Find the last word of this sentence in the timestamps
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
# VECTOR 3: Semantic-Acoustic Alignment
# "The Emotion Detector"
# ─────────────────────────────────────────────

LABEL_TO_SCORE = {
    "positive": 1.0,
    "neutral":  0.0,
    "negative": -1.0,
    # cardiffnlp model uses these label names:
    "label_0":  -1.0,   # negative
    "label_1":   0.0,   # neutral
    "label_2":   1.0,   # positive
}

def extract_sentiment_features(transcript: str) -> dict:
    """
    Runs RoBERTa sentiment analysis on the transcript.

    This score is NOT used alone. It becomes meaningful in Phase 3
    when we cross-reference it against actual pitch and energy.
    A highly negative/emotional transcript with completely flat pitch
    is a strong deepfake indicator.

    Returns:
        sentiment_label : 'positive', 'neutral', or 'negative'
        sentiment_score : confidence of the prediction (0.0 to 1.0)
        sentiment_value : numeric mapping (-1.0, 0.0, or 1.0) for
                          easy comparison against acoustic features
    """
    if not transcript or transcript.strip() == "":
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_value": 0.0
        }

    try:
        result = sentiment_analyzer(transcript[:512])[0]  # truncate to model max
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
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_value": 0.0
        }


# ─────────────────────────────────────────────
# MASTER FUNCTION: Run All of Phase 2
# ─────────────────────────────────────────────

def build_phase2_features(
    phase1_csv: str = "/content/processed/phase1_metadata.csv",
    output_csv: str = "/content/processed/phase2_features.csv"
) -> pd.DataFrame:
    """
    Reads the Phase 1 CSV, runs all three linguistic vectors on every row,
    and saves an enriched CSV with all new feature columns appended.
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

    # Merge all feature dicts into the dataframe
    df_lexical   = pd.DataFrame(lexical_records)
    df_prosodic  = pd.DataFrame(prosodic_records)
    df_sentiment = pd.DataFrame(sentiment_records)

    df_out = pd.concat([df, df_lexical, df_prosodic, df_sentiment], axis=1)
    df_out.to_csv(output_csv, index=False)

    print(f"\n✅ Phase 2 complete. Saved to: {output_csv}")
    print(f"   Total features per chunk: {len(df_out.columns)} columns")
    return df_out