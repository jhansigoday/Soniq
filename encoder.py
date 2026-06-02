"""
encoder.py  —  Layer 3: Semantic Fusion Encoder

Combines:
  • Overlapping 10-second windows with 2-second crossfade
    → eliminates boundary artefacts on long recordings
  • Feature fusion: HuBERT (768) + energy (1) + articulation (1) + pitch (1)
    → 771-dimensional expressive embeddings at ~50 fps
  • All 94M HuBERT params frozen (inference-only)
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf

from scipy.ndimage import gaussian_filter1d
from transformers import HubertModel, Wav2Vec2FeatureExtractor


# ── Window parameters ─────────────────────────────────────────────────────────
_WINDOW_S  = 10.0    # seconds per processing window
_OVERLAP_S =  2.0    # crossfade overlap between adjacent windows
_HUBERT_STRIDE = 320  # HuBERT conv stride (samples at 16kHz → 1 frame)


# ============================================================
# Expressive Feature Extractors  (from your uploaded version)
# ============================================================

def compute_energy(audio: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=320,
    )[0]
    return rms.astype(np.float32)


def compute_pitch(audio: np.ndarray, sr: int) -> np.ndarray:
    f0, _, _ = librosa.pyin(
        audio,
        fmin=65,
        fmax=1200,
        sr=sr,
    )
    f0 = np.nan_to_num(f0)
    f0 = gaussian_filter1d(f0, sigma=1)
    return f0.astype(np.float32)


def compute_articulation(audio: np.ndarray) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr = gaussian_filter1d(zcr, sigma=1)
    return zcr.astype(np.float32)


# ============================================================
# Crossfade Merge  (from overlapping-window version)
# ============================================================

def _crossfade_merge(chunks: list, overlap_frames: int) -> np.ndarray:
    """
    Merge HuBERT embedding chunks with linear crossfade in the overlap
    region to eliminate discontinuities at window boundaries.
    Each chunk has shape [T_i, D].
    """
    if len(chunks) == 1:
        return chunks[0]

    result_parts = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            # Keep everything except the last overlap_frames
            result_parts.append(chunk[:-overlap_frames])

        elif i == len(chunks) - 1:
            # Blend overlap then keep remainder
            prev_tail = chunks[i - 1][-overlap_frames:]
            cur_head  = chunk[:overlap_frames]
            fade_out  = np.linspace(1.0, 0.0, overlap_frames)[:, None]
            fade_in   = np.linspace(0.0, 1.0, overlap_frames)[:, None]
            blended   = prev_tail * fade_out + cur_head * fade_in
            result_parts.append(blended)
            result_parts.append(chunk[overlap_frames:])

        else:
            prev_tail = chunks[i - 1][-overlap_frames:]
            cur_head  = chunk[:overlap_frames]
            fade_out  = np.linspace(1.0, 0.0, overlap_frames)[:, None]
            fade_in   = np.linspace(0.0, 1.0, overlap_frames)[:, None]
            blended   = prev_tail * fade_out + cur_head * fade_in
            result_parts.append(blended)
            result_parts.append(chunk[overlap_frames:-overlap_frames])

    return np.concatenate(result_parts, axis=0)


# ============================================================
# Main Encoding
# ============================================================

def run_encoding(vocals_path: str, output_path: str) -> str:

    print("\n============================================================")
    print("  LAYER 3 — Semantic Fusion Encoder")
    print("  (HuBERT + energy + pitch + articulation  |  10s windows)")
    print("============================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load HuBERT ───────────────────────────────────────────────────────────
    print("[Layer 3] Loading HuBERT Base  (facebook/hubert-base-ls960) ...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"[Layer 3] Device: {device}  |  94M params frozen")

    # ── Load audio ────────────────────────────────────────────────────────────
    audio, sr = sf.read(vocals_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    total_s = len(audio) / sr
    print(f"[Layer 3] Loaded: {vocals_path}  ({total_s:.1f}s  sr={sr})")

    # ── Expressive features at original SR ───────────────────────────────────
    print("[Layer 3] Extracting expressive features (energy / pitch / articulation) ...")
    energy       = compute_energy(audio)           # [T_e]
    pitch        = compute_pitch(audio, sr)        # [T_p]
    articulation = compute_articulation(audio)     # [T_a]

    # ── Resample to 16 kHz for HuBERT ────────────────────────────────────────
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio
    sr_16k = 16000

    print(f"[Layer 3] Audio at 16 kHz  |  duration={len(audio_16k)/sr_16k:.1f}s")

    # ── Overlapping window HuBERT inference ──────────────────────────────────
    window_samples  = int(_WINDOW_S  * sr_16k)   # 160000 samples
    overlap_samples = int(_OVERLAP_S * sr_16k)   #  32000 samples
    step_samples    = window_samples - overlap_samples

    chunks  = []
    start   = 0
    win_no  = 0
    total_windows = max(1, int(
        np.ceil((len(audio_16k) - overlap_samples) / step_samples)
    ))

    while start < len(audio_16k):
        end     = min(start + window_samples, len(audio_16k))
        segment = audio_16k[start:end]
        win_no += 1

        inputs      = processor(segment, sampling_rate=16000, return_tensors="pt")
        input_vals  = inputs.input_values.to(device)

        with torch.no_grad():
            out        = model(input_vals)
            emb        = out.last_hidden_state      # [1, T, 768]

        emb_np = emb.squeeze(0).cpu().numpy()       # [T, 768]
        chunks.append(emb_np)

        print(f"[Layer 3]   Window {win_no}/{total_windows}  "
              f"samples [{start}:{end}]  →  embeddings {emb_np.shape}")

        if end == len(audio_16k):
            break
        start += step_samples

    # ── Crossfade merge ───────────────────────────────────────────────────────
    if len(chunks) > 1:
        overlap_frames = max(1, overlap_samples // _HUBERT_STRIDE)
        semantic_embeddings = _crossfade_merge(chunks, overlap_frames)
        print(f"[Layer 3] Crossfade merged {len(chunks)} windows  "
              f"→  {semantic_embeddings.shape}")
    else:
        semantic_embeddings = chunks[0]

    # ── Align expressive features to HuBERT frame count ──────────────────────
    target_frames = semantic_embeddings.shape[0]

    energy       = librosa.util.fix_length(energy,       size=target_frames)
    articulation = librosa.util.fix_length(articulation, size=target_frames)
    pitch        = librosa.util.fix_length(pitch,        size=target_frames)

    # ── Feature fusion: [T, 768] + [T,1] + [T,1] + [T,1] → [T, 771] ─────────
    print("[Layer 3] Fusing HuBERT embeddings with expressive features ...")
    fused_embeddings = np.concatenate([
        semantic_embeddings,        # 768 dims  — phonetic + timbral
        energy      [:, None],      #   1 dim   — loudness envelope
        articulation[:, None],      #   1 dim   — zero-crossing rate
        pitch       [:, None],      #   1 dim   — fundamental frequency
    ], axis=1).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, fused_embeddings)

    print(f"[Layer 3] ✅ Saved embeddings: {output_path}")
    print(f"[Layer 3]    Shape: {fused_embeddings.shape}  "
          f"({fused_embeddings.shape[0]} frames @ ~50 fps  =  "
          f"~{fused_embeddings.shape[0]/50:.1f}s)")
    print(f"[Layer 3]    Dims:  768 HuBERT  +  1 energy  "
          f"+  1 articulation  +  1 pitch  =  771 total")

    return output_path


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    run_encoding(
        "soniq_outputs/conditioned_vocals.wav",
        "soniq_outputs/hubert_embeddings.npy",
    )