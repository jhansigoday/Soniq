"""
reference_generator.py  —  Instrument Reference Audio Manager

Ensures a reference WAV exists for the requested instrument.
Called by orchestrator.py (pre-flight) and converter.py (Layer 4).

Priority order:
  1. Use the path from config.yaml instruments.<name>.reference if the file exists
  2. If the file is missing, synthesise a basic chromatic reference using
     scipy so the pipeline can still run (lower quality than a real recording)
"""

import os
import numpy as np
import soundfile as sf
import yaml


TARGET_SR = 44100

# Notes to include in a synthesised reference (chromatic, mid-register)
_SYNTH_NOTES_HZ = [
    130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,  # C3–B3
    261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,  # C4–B4
    523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,  # C5–B5
]

# Per-instrument harmonic profiles (fundamental + partials)
_HARMONIC_PROFILES = {
    'piano':  [1.0, 0.62, 0.42, 0.28, 0.20, 0.14, 0.10, 0.07, 0.05, 0.03],
    'violin': [1.0, 0.75, 0.55, 0.45, 0.35, 0.28, 0.22, 0.17, 0.12, 0.09],
    'cello':  [1.0, 0.85, 0.65, 0.55, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08],
    'flute':  [1.0, 0.28, 0.07, 0.03, 0.015, 0.008, 0.004, 0.002],
    'guitar': [1.0, 0.55, 0.38, 0.28, 0.22, 0.17, 0.12, 0.09, 0.07, 0.05],
    'synth':  [1.0, 0.5,  0.33, 0.25, 0.2,  0.167, 0.143, 0.125],
    'trumpet': [1.0, 0.90, 0.78, 0.58, 0.38, 0.22, 0.13, 0.08, 0.05, 0.03],
}

# Per-instrument decay time in seconds
_DECAY = {
    'piano':  2.2,
    'violin': 0.5,
    'cello':  0.6,
    'flute':  0.4,
    'guitar': 1.0,
    'synth':  0.3,
    'trumpet': 0.5,
}

# Per-instrument attack time in seconds
_ATTACK = {
    'piano':  0.003,
    'violin': 0.045,
    'cello':  0.07,
    'flute':  0.025,
    'guitar': 0.004,
    'synth':  0.018,
    'trumpet': 0.015,
}


def _synthesise_reference(instrument: str, out_path: str,
                           duration_per_note: float = 1.4,
                           target_sr: int = TARGET_SR) -> str:
    """
    Synthesise a basic chromatic reference WAV for the given instrument.
    Used only when no real recording is available. Quality is acceptable
    for pipeline testing but a real recording will always sound better.
    """
    profile = _HARMONIC_PROFILES.get(instrument, _HARMONIC_PROFILES['piano'])
    decay_s = _DECAY.get(instrument, 1.0)
    atk_s   = _ATTACK.get(instrument, 0.01)

    segments = []
    for f0 in _SYNTH_NOTES_HZ:
        n = int(duration_per_note * target_sr)
        t = np.arange(n, dtype=np.float64) / target_sr

        # Build harmonic tone
        tone = np.zeros(n, dtype=np.float64)
        for h_idx, amp in enumerate(profile):
            h    = h_idx + 1
            freq = min(f0 * h, target_sr / 2.0 - 1.0)
            tone += amp * np.sin(2.0 * np.pi * freq * t)
        tone /= (np.sum(profile) + 1e-8)

        # Attack + decay envelope
        env        = np.ones(n, dtype=np.float64)
        atk_samp   = max(1, int(atk_s * target_sr))
        env[:atk_samp] = np.linspace(0.0, 1.0, atk_samp) ** 0.3
        env       *= np.exp(-t / (decay_s / np.log(2.0)))

        segments.append((tone * env).astype(np.float32))

    # 20 ms crossfade between notes
    fade = int(0.02 * target_sr)
    result = segments[0]
    for seg in segments[1:]:
        if len(result) >= fade and len(seg) >= fade:
            fo = np.linspace(1, 0, fade, dtype=np.float32)
            fi = np.linspace(0, 1, fade, dtype=np.float32)
            result[-fade:] = result[-fade:] * fo + seg[:fade] * fi
            result = np.concatenate([result, seg[fade:]])
        else:
            result = np.concatenate([result, seg])

    # Trim / pad to 30 seconds
    target_len = int(30.0 * target_sr)
    if len(result) > target_len:
        result = result[:target_len]
    elif len(result) < target_len:
        result = np.pad(result, (0, target_len - len(result)))

    # Normalise
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.9

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sf.write(out_path, result, target_sr, subtype='PCM_24')
    print(f"[ReferenceGen] ✅ Synthesised reference saved: {out_path}  (30s)")
    return out_path


def ensure_reference(instrument: str, config_path: str = 'config.yaml') -> str:
    """
    Return path to a valid reference WAV for the instrument.

    1. Reads config.yaml → instruments.<instrument>.reference
    2. If that file exists → return it immediately
    3. If missing → synthesise a basic reference and save it there
    """
    # Load config
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}

    ref_path = (cfg
                .get('instruments', {})
                .get(instrument, {})
                .get('reference', f'instruments/{instrument}/reference.wav'))

    # Case 1: real reference exists — use it
    if os.path.isfile(ref_path):
        size_kb = os.path.getsize(ref_path) / 1024
        print(f"[ReferenceGen] Reference found: {ref_path}  ({size_kb:.0f} KB)")
        return ref_path

    # Case 2: missing — synthesise
    print(f"[ReferenceGen] ⚠️  Reference not found at '{ref_path}'")
    print(f"[ReferenceGen]    Synthesising basic {instrument} reference ...")
    print(f"[ReferenceGen]    (For best quality, replace with a real recording)")
    return _synthesise_reference(instrument, ref_path)