# ═══════════════════════════════════════════════════════════════
# feature_extractor.py
# Phase 3: Targeted Acoustic Feature Extraction
#
# Extracts three layers of acoustic features:
#   1. Global Acoustics  — Mel-spectrogram statistics (baseline)
#   2. Targeted Pitch    — F0 contour using parselmouth/praat
#   3. Targeted Energy   — RMS energy contour using librosa
#
# These features are later cross-referenced with Phase 2 sentiment
# scores in the fusion layer (Phase 4) to detect emotion-acoustic
# mismatches — a strong deepfake indicator.
# ═══════════════════════════════════════════════════════════════

import os
import ast
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE    = 16000
N_MELS         = 128       # number of mel filterbanks
HOP_LENGTH     = 512       # samples between frames
N_FFT          = 1024      # FFT window size
PITCH_FLOOR    = 75.0      # Hz — lowest expected human pitch (male bass)
PITCH_CEILING  = 400.0     # Hz — highest expected human pitch (female soprano)


# ═══════════════════════════════════════════════════════════════
# LAYER 1: Global Acoustics — Mel Spectrogram Statistics
# ═══════════════════════════════════════════════════════════════

def extract_mel_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Computes the Mel-spectrogram of the audio chunk and summarizes it
    into statistics. This acts as the acoustic baseline — it captures
    overall tonal and frequency patterns across the whole chunk.

    A Mel-spectrogram converts raw audio into a 2D image of frequency
    content over time, scaled to the Mel scale (which mirrors how human
    ears perceive pitch — logarithmically, not linearly).

    Rather than feeding the raw 2D spectrogram into the network here,
    we extract summary statistics per Mel band. This gives the model
    a compact, fixed-size acoustic fingerprint regardless of chunk length.

    Returns:
        mel_mean      : mean energy across all Mel bands and time frames
        mel_std       : standard deviation — how much energy varies
        mel_min       : minimum energy value in the spectrogram
        mel_max       : maximum energy value
        mel_bandwidth : mel_max - mel_min, measures dynamic range
                        AI voices tend to have narrower dynamic range
        mel_flatness  : how evenly distributed energy is across bands
                        a flat spectrum suggests synthetic/processed audio
    """
    if len(audio) == 0:
        return {
            "mel_mean": 0.0, "mel_std": 0.0, "mel_min": 0.0,
            "mel_max": 0.0, "mel_bandwidth": 0.0, "mel_flatness": 0.0
        }

    # Compute Mel spectrogram and convert to decibels
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Spectral flatness — closer to 1.0 means more noise-like/synthetic
    flatness = librosa.feature.spectral_flatness(
        y=audio, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    return {
        "mel_mean":      round(float(np.mean(mel_db)), 4),
        "mel_std":       round(float(np.std(mel_db)), 4),
        "mel_min":       round(float(np.min(mel_db)), 4),
        "mel_max":       round(float(np.max(mel_db)), 4),
        "mel_bandwidth": round(float(np.max(mel_db) - np.min(mel_db)), 4),
        "mel_flatness":  round(float(np.mean(flatness)), 6),
    }


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Targeted Pitch (F0) Extraction
# ═══════════════════════════════════════════════════════════════

def extract_pitch_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extracts the fundamental frequency (F0) contour using Praat via
    parselmouth. F0 is the perceived 'pitch' of the voice.

    Why Praat and not just librosa?
    Praat's pitch extraction algorithm (SHR - Subharmonic-to-Harmonic
    Ratio) is the gold standard in phonetics research. It handles
    voiced/unvoiced transitions and pitch tracking far more accurately
    than FFT-based methods — critical for detecting subtle AI artifacts.

    Key deepfake signals in pitch:
    - Unnaturally low std (pitch barely moves — robotic flatness)
    - Very low pitch_range (no emotional highs and lows)
    - Low pitch_slope_std (pitch changes are uniform, not organic)
    - High pitch_on_filler: humans waver in pitch on 'um', 'uh';
      AI pitch stays flat even on these words

    Returns:
        pitch_mean          : average F0 in Hz across voiced frames
        pitch_std           : standard deviation of F0 — key naturalness indicator
        pitch_min           : lowest pitch detected
        pitch_max           : highest pitch detected
        pitch_range         : pitch_max - pitch_min
        pitch_slope_mean    : mean rate of pitch change frame-to-frame
        pitch_slope_std     : std of pitch change — humans are irregular
        voiced_fraction     : proportion of frames that are voiced (0.0-1.0)
                              AI sometimes has unnaturally high voiced fraction
        pitch_on_filler     : std of pitch specifically during filler words
                              low value = AI (flat on fillers), high = human
    """
    empty = {
        "pitch_mean": 0.0, "pitch_std": 0.0, "pitch_min": 0.0,
        "pitch_max": 0.0, "pitch_range": 0.0,
        "pitch_slope_mean": 0.0, "pitch_slope_std": 0.0,
        "voiced_fraction": 0.0, "pitch_on_filler": 0.0
    }

    if len(audio) == 0:
        return empty

    try:
        # parselmouth needs a Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch_obj = sound.to_pitch(
            pitch_floor=PITCH_FLOOR,
            pitch_ceiling=PITCH_CEILING
        )

        # Extract all F0 values frame by frame
        pitch_values = pitch_obj.selected_array["frequency"]

        # Separate voiced frames (F0 > 0) from unvoiced (F0 == 0)
        voiced = pitch_values[pitch_values > 0]
        total_frames = len(pitch_values)

        if len(voiced) < 2:
            return empty

        voiced_fraction = len(voiced) / total_frames if total_frames > 0 else 0.0

        # Frame-to-frame pitch change (slope) — measures naturalness of movement
        pitch_slopes = np.diff(voiced)

        return {
            "pitch_mean":       round(float(np.mean(voiced)), 4),
            "pitch_std":        round(float(np.std(voiced)), 4),
            "pitch_min":        round(float(np.min(voiced)), 4),
            "pitch_max":        round(float(np.max(voiced)), 4),
            "pitch_range":      round(float(np.max(voiced) - np.min(voiced)), 4),
            "pitch_slope_mean": round(float(np.mean(pitch_slopes)), 4),
            "pitch_slope_std":  round(float(np.std(pitch_slopes)), 4),
            "voiced_fraction":  round(float(voiced_fraction), 4),
            "pitch_on_filler":  0.0,   # filled in by extract_filler_pitch() below
        }

    except Exception as e:
        print(f"  [WARN] Pitch extraction failed: {e}")
        return empty


def extract_filler_pitch(
    audio: np.ndarray,
    word_timestamps: list,
    sr: int = SAMPLE_RATE
) -> float:
    """
    Extracts pitch SPECIFICALLY during filler words (um, uh, hmm).

    This is one of the strongest deepfake signals in the whole pipeline.
    When a human says 'ummmm', their pitch naturally rises and falls —
    it is a thinking sound with real vocal variation.
    When AI generates 'um', it produces a flat, tonally uniform sound
    because it has no cognitive state to express.

    Returns the std of F0 values during filler word segments.
    Low value (near 0) = AI-like flat filler.
    High value = human-like variable filler.
    """
    FILLER_SET = {"um", "uh", "hmm", "ah", "er", "uhh", "umm"}

    if not word_timestamps or len(audio) == 0:
        return 0.0

    filler_pitch_values = []

    for wt in word_timestamps:
        if wt["word"].strip().lower() not in FILLER_SET:
            continue

        # Slice the audio to just the filler word segment
        start_sample = int(wt["start"] * sr)
        end_sample   = int(wt["end"]   * sr)
        segment      = audio[start_sample:end_sample]

        if len(segment) < 100:   # too short to analyze
            continue

        try:
            sound     = parselmouth.Sound(segment, sampling_frequency=sr)
            pitch_obj = sound.to_pitch(
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING
            )
            vals   = pitch_obj.selected_array["frequency"]
            voiced = vals[vals > 0]
            if len(voiced) > 1:
                filler_pitch_values.extend(voiced.tolist())
        except Exception:
            continue

    if len(filler_pitch_values) < 2:
        return 0.0

    return round(float(np.std(filler_pitch_values)), 4)


# ═══════════════════════════════════════════════════════════════
# LAYER 3: Targeted Energy (RMS) Extraction
# ═══════════════════════════════════════════════════════════════

def extract_energy_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extracts RMS (Root Mean Square) energy contour using librosa.
    RMS energy is the perceived loudness/volume of the voice over time.

    Key deepfake signals in energy:
    - Low energy_std: AI voices maintain unnaturally consistent volume
    - Low energy_range: no dynamic loudness variation
    - Low energy_slope_std: volume changes are robotic and uniform
    - High energy_flatness: energy is distributed too evenly over time

    The cross-reference with Phase 2 sentiment:
    If sentiment_value is -1.0 (very negative/angry text) but
    energy_std is near 0 (completely flat volume), that mismatch
    is a strong deepfake signal — angry words with calm robotic delivery.

    Returns:
        energy_mean       : average loudness across the chunk
        energy_std        : std of loudness — key naturalness indicator
        energy_min        : quietest moment
        energy_max        : loudest moment
        energy_range      : energy_max - energy_min
        energy_slope_mean : mean rate of volume change
        energy_slope_std  : std of volume change — humans are irregular
        energy_flatness   : how evenly distributed energy is over time
                            high value suggests synthetic processing
    """
    empty = {
        "energy_mean": 0.0, "energy_std": 0.0, "energy_min": 0.0,
        "energy_max": 0.0, "energy_range": 0.0,
        "energy_slope_mean": 0.0, "energy_slope_std": 0.0,
        "energy_flatness": 0.0
    }

    if len(audio) == 0:
        return empty

    try:
        rms = librosa.feature.rms(
            y=audio, hop_length=HOP_LENGTH, frame_length=N_FFT
        )[0]

        if len(rms) < 2:
            return empty

        rms_slopes = np.diff(rms)

        # Energy flatness over time — std of a rolling window mean
        window_size = max(1, len(rms) // 10)
        windows     = [
            np.mean(rms[i:i + window_size])
            for i in range(0, len(rms) - window_size, window_size)
        ]
        energy_flatness = float(np.std(windows)) if len(windows) > 1 else 0.0

        return {
            "energy_mean":       round(float(np.mean(rms)), 6),
            "energy_std":        round(float(np.std(rms)), 6),
            "energy_min":        round(float(np.min(rms)), 6),
            "energy_max":        round(float(np.max(rms)), 6),
            "energy_range":      round(float(np.max(rms) - np.min(rms)), 6),
            "energy_slope_mean": round(float(np.mean(rms_slopes)), 6),
            "energy_slope_std":  round(float(np.std(rms_slopes)), 6),
            "energy_flatness":   round(energy_flatness, 6),
        }

    except Exception as e:
        print(f"  [WARN] Energy extraction failed: {e}")
        return empty


# ═══════════════════════════════════════════════════════════════
# THE FUSION: Semantic-Acoustic Mismatch Score
# ═══════════════════════════════════════════════════════════════

def compute_mismatch_score(
    sentiment_value: float,
    pitch_std: float,
    energy_std: float
) -> dict:
    """
    Cross-references Phase 2 sentiment against Phase 3 acoustics.
    This is the core insight of the whole pipeline.

    Logic:
    - If sentiment is strongly emotional (abs value close to 1.0)
      but pitch_std and energy_std are both near 0 (flat, robotic),
      the mismatch is high → likely deepfake.
    - If sentiment is neutral AND acoustics are flat → no mismatch,
      could still be real (someone speaking monotonously).
    - If sentiment is emotional AND acoustics show variation → human.

    mismatch_score ranges from 0.0 (no mismatch) to 1.0 (full mismatch).

    Returns:
        emotion_intensity    : how strongly emotional the text is (0.0-1.0)
        acoustic_flatness    : how flat the acoustics are (0.0-1.0, normalized)
        mismatch_score       : emotion_intensity * acoustic_flatness
    """
    # How emotionally charged is the text? (0=neutral, 1=very emotional)
    emotion_intensity = abs(float(sentiment_value))

    # Normalize pitch_std and energy_std to 0-1 scale
    # These thresholds are based on typical human speech ranges
    PITCH_STD_MAX  = 50.0   # Hz — above this is very expressive speech
    ENERGY_STD_MAX = 0.05   # RMS units — above this is very dynamic speech

    pitch_flatness  = max(0.0, 1.0 - (pitch_std  / PITCH_STD_MAX))
    energy_flatness = max(0.0, 1.0 - (energy_std / ENERGY_STD_MAX))

    # Average the two flatness measures
    acoustic_flatness = (pitch_flatness + energy_flatness) / 2.0

    # Mismatch = emotional text × flat acoustics
    mismatch_score = emotion_intensity * acoustic_flatness

    return {
        "emotion_intensity":  round(emotion_intensity, 4),
        "acoustic_flatness":  round(acoustic_flatness, 4),
        "mismatch_score":     round(mismatch_score, 4),
    }


# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTION: Run All of Phase 3
# ═══════════════════════════════════════════════════════════════

def build_phase3_features(
    phase2_csv: str = "/content/processed/phase2_features.csv",
    output_csv: str = "/content/processed/phase3_features.csv"
) -> pd.DataFrame:
    """
    Reads Phase 2 CSV, loads each audio chunk, runs all three acoustic
    layers plus the mismatch score, appends results, saves new CSV.
    """
    print("Loading Phase 2 data...")
    df = pd.read_csv(phase2_csv)
    print(f"  {len(df)} chunks loaded.\n")

    mel_records      = []
    pitch_records    = []
    energy_records   = []
    mismatch_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phase 3 Progress"):
        chunk_file = row.get("chunk_file", "")

        # ── Load audio chunk from disk ──
        if not chunk_file or not os.path.exists(str(chunk_file)):
            print(f"  [WARN] Missing chunk file: {chunk_file}")
            mel_records.append({
                "mel_mean":0.0,"mel_std":0.0,"mel_min":0.0,
                "mel_max":0.0,"mel_bandwidth":0.0,"mel_flatness":0.0
            })
            pitch_records.append({
                "pitch_mean":0.0,"pitch_std":0.0,"pitch_min":0.0,
                "pitch_max":0.0,"pitch_range":0.0,"pitch_slope_mean":0.0,
                "pitch_slope_std":0.0,"voiced_fraction":0.0,"pitch_on_filler":0.0
            })
            energy_records.append({
                "energy_mean":0.0,"energy_std":0.0,"energy_min":0.0,
                "energy_max":0.0,"energy_range":0.0,"energy_slope_mean":0.0,
                "energy_slope_std":0.0,"energy_flatness":0.0
            })
            mismatch_records.append({
                "emotion_intensity":0.0,"acoustic_flatness":0.0,"mismatch_score":0.0
            })
            continue

        try:
            audio, sr = librosa.load(str(chunk_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  [ERROR] Cannot load {chunk_file}: {e}")
            continue

        # ── Parse word timestamps for filler pitch analysis ──
        word_timestamps = []
        raw_wt = row.get("word_timestamps", "[]")
        if raw_wt and str(raw_wt).strip() not in ("", "[]", "nan"):
            try:
                word_timestamps = ast.literal_eval(str(raw_wt))
            except Exception:
                pass

        # ── Run all three acoustic layers ──
        mel_feats   = extract_mel_features(audio, sr)
        pitch_feats = extract_pitch_features(audio, sr)
        energy_feats = extract_energy_features(audio, sr)

        # ── Fill in filler-specific pitch std ──
        filler_p = extract_filler_pitch(audio, word_timestamps, sr)
        pitch_feats["pitch_on_filler"] = filler_p

        # ── Compute semantic-acoustic mismatch ──
        sentiment_value = float(row.get("sentiment_value", 0.0))
        mismatch = compute_mismatch_score(
            sentiment_value,
            pitch_feats["pitch_std"],
            energy_feats["energy_std"]
        )

        mel_records.append(mel_feats)
        pitch_records.append(pitch_feats)
        energy_records.append(energy_feats)
        mismatch_records.append(mismatch)

    # ── Merge everything into one DataFrame ──
    df_out = pd.concat([
        df,
        pd.DataFrame(mel_records),
        pd.DataFrame(pitch_records),
        pd.DataFrame(energy_records),
        pd.DataFrame(mismatch_records),
    ], axis=1)

    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Phase 3 complete. {len(df_out.columns)} total columns.")
    print(f"   Saved to: {output_csv}")
    return df_out


if __name__ == "__main__":
    df = build_phase3_features()
    print(df.head())