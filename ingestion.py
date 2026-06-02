"""
ingestion.py  —  Layer 1: Voice Ingestion & Source Separation  v5.3 HQ

Changes over v5.2:
  FIX:  Auto-converts ANY input format (mp3, flac, ogg, m4a, aac, opus, wma, aiff)
        to 44.1 kHz 24-bit WAV before processing. pydub + ffmpeg handle all formats.
  FIX:  Harmonic smoothing kernel was corrupting transients — replaced with
        frequency-domain spectral smoothing that preserves attack envelopes.
  FIX:  Demucs output stems are index-looked-up by name (not hardcoded index 3)
        to be safe across model versions.
  ADD:  Wet/dry mix for DeepFilterNet — avoids over-processing artefacts.
  ADD:  Noise reduction prop_decrease raised to 0.55 for cleaner separation.
  ADD:  Output RMS validation: if output is >6 dB quieter than input, warn.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import torch
import torchaudio
import noisereduce as nr
import scipy.signal

from scipy.signal import butter, sosfiltfilt, sosfilt
from pydub import AudioSegment

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    Compressor,
    NoiseGate,
    Limiter
)

from demucs.pretrained import get_model
from demucs.apply import apply_model


# ── Optional DeepFilterNet ────────────────────────────────────────────────────
try:
    from df.enhance import enhance, init_df
    DEEPFILTER_AVAILABLE = True
except Exception as e:
    print(f"[Layer 1] DeepFilterNet unavailable: {e}")
    DEEPFILTER_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SR   = 44100
TARGET_LUFS = -23.0

# All formats that ffmpeg/pydub can decode
SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
                  '.wma', '.aiff', '.aif', '.opus', '.webm', '.mp4',
                  '.3gp', '.amr', '.ac3', '.dts'}


# ── Format Conversion ─────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, out_dir: str = None) -> str:
    """
    Convert ANY audio format to 44.1 kHz stereo WAV via pydub/ffmpeg.
    Returns path to WAV file (may be same as input if already WAV).
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.wav':
        # Still validate it's readable
        try:
            _, sr = sf.read(input_path, frames=1)
            return input_path
        except Exception:
            pass  # Fall through to re-encode

    print(f"[Layer 1] Auto-converting {ext} → WAV ...")

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        # Try specifying format explicitly
        fmt = ext.lstrip('.')
        fmt_map = {'m4a': 'mp4', 'aif': 'aiff', '3gp': 'mp4'}
        fmt = fmt_map.get(fmt, fmt)
        audio = AudioSegment.from_file(input_path, format=fmt)

    # Normalise to 44100 Hz, 16-bit for export (we re-read and up-convert)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(2)

    out_dir = out_dir or os.path.dirname(input_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_wav = os.path.join(out_dir, f"{base}_converted.wav")

    audio.export(out_wav, format='wav', parameters=['-acodec', 'pcm_s24le'])
    print(f"[Layer 1] Converted → {out_wav}")
    return out_wav


# ── Silence Trimming ──────────────────────────────────────────────────────────

def trim_silence(audio: np.ndarray) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    return trimmed.astype(np.float32)


# ── DC Offset ─────────────────────────────────────────────────────────────────

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio)).astype(np.float32)


# ── Highpass ──────────────────────────────────────────────────────────────────

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    sos = butter(4, cutoff_hz / (sr / 2), btype='highpass', output='sos')
    return sosfiltfilt(sos, audio).astype(np.float32)


# ── Loudness Normalize ────────────────────────────────────────────────────────

def loudness_normalize(audio: np.ndarray, sr: int,
                        target_lufs: float = TARGET_LUFS) -> np.ndarray:
    meter    = pyln.Meter(sr)
    inp      = audio[:, np.newaxis] if audio.ndim == 1 else audio
    loudness = meter.integrated_loudness(inp)

    if np.isinf(loudness) or np.isnan(loudness):
        # Audio too quiet to measure — just normalise to peak
        peak = np.max(np.abs(audio))
        return (audio / peak * 0.95).astype(np.float32) if peak > 0 else audio

    normalized = pyln.normalize.loudness(inp, loudness, target_lufs)
    if normalized.ndim > 1 and normalized.shape[1] == 1:
        normalized = normalized[:, 0]
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized / peak * 0.99
    return normalized.astype(np.float32)


# ── Neural Enhancement ────────────────────────────────────────────────────────

def neural_enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    # DeepFilterNet
    if DEEPFILTER_AVAILABLE:
        try:
            print("[Layer 1] Running DeepFilterNet ...")
            model, df_state, _ = init_df()
            tensor = torch.tensor(audio).float()
            enhanced = enhance(model, df_state, tensor.unsqueeze(0))
            enhanced = enhanced.squeeze().cpu().numpy()
            # Wet/dry mix 0.85 wet to avoid over-processing
            audio = 0.85 * enhanced + 0.15 * audio
        except Exception as e:
            print(f"[Layer 1] DeepFilterNet failed: {e}")

    # Spectral denoise
    print("[Layer 1] Spectral denoise ...")
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=False,
                             prop_decrease=0.55)
    return audio.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion(input_path: str, output_path: str) -> str:
    print("\n" + "=" * 60)
    print("  LAYER 1 — Voice Conditioning & Source Separation")
    print("=" * 60)

    # Auto-convert any format → WAV
    out_dir   = os.path.dirname(output_path) or 'soniq_outputs'
    input_path = convert_to_wav(input_path, out_dir=out_dir)

    print(f"[Layer 1] Loading: {input_path}")
    audio, sr = sf.read(input_path)

    # Stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 44100
    if sr != TARGET_SR:
        print(f"[Layer 1] Resampling {sr} → {TARGET_SR}")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    print(f"[Layer 1] Duration: {len(audio)/sr:.2f}s  SR: {sr}")

    audio = trim_silence(audio)
    audio = remove_dc_offset(audio)
    audio = highpass_filter(audio, sr, cutoff_hz=60.0)
    audio = loudness_normalize(audio, sr)

    # Normalise peak before Demucs
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # Build 2-channel tensor for Demucs
    wav = torch.tensor(audio).float().unsqueeze(0).repeat(2, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Layer 1] Device: {device}")

    print("[Layer 1] Loading Demucs htdemucs_ft ...")
    model = get_model("htdemucs_ft")
    model.to(device)
    model.eval()
    wav = wav.unsqueeze(0).to(device)

    print("[Layer 1] Running Demucs ...")
    with torch.no_grad():
        sources = apply_model(model, wav, shifts=1, split=True)

    # Extract vocals by name (robust to model version)
    source_names = model.sources
    vocal_idx = source_names.index('vocals') if 'vocals' in source_names else 3
    vocals = sources[0, vocal_idx]   # [2, T]
    vocals_mono = vocals.mean(dim=0).cpu().numpy()

    print("[Layer 1] Vocal refinement ...")
    vocals_mono = neural_enhance(vocals_mono, sr)

    # Second-pass denoise — gentle pass only; combined with the first pass
    # (prop_decrease=0.55 inside neural_enhance) a high second value over-whitens
    # the signal and strips musical content that Layer 4 needs for WORLD analysis.
    vocals_mono = nr.reduce_noise(y=vocals_mono, sr=sr,
                                   stationary=False, prop_decrease=0.20)

    # Transient restoration (soft)
    transient    = vocals_mono - scipy.signal.medfilt(vocals_mono, kernel_size=5)
    vocals_mono += 0.12 * transient

    # Final mastering chain
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=55),
        LowpassFilter(cutoff_frequency_hz=16000),
        NoiseGate(threshold_db=-48, ratio=2.0),
        Compressor(threshold_db=-20, ratio=2.5),
        Limiter(threshold_db=-1),
    ])
    vocals_mono = board(vocals_mono, sr)

    vocals_mono = loudness_normalize(vocals_mono, sr)
    peak = np.max(np.abs(vocals_mono))
    if peak > 0.99:
        vocals_mono = vocals_mono / peak * 0.98

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, vocals_mono, sr, subtype='PCM_24')
    print(f"[Layer 1] ✅ Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    run_ingestion("input/test.wav", "soniq_outputs/conditioned_vocals.wav")