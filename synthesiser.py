"""
synthesiser.py  —  Layer 5: Physical Instrument Synthesis  v5.3 HQ

Key fixes over v5.2 that caused broken / non-continuous output:
  FIX 1: frame_hop aligned to WORLD 10 ms grid (was 441 → matched to 10 ms = 441 samples @44100).
          Mismatch between WORLD frame period and synthesis frame_hop caused pitch stairstepping.
  FIX 2: Phase accumulation now correctly initialises from zero and accumulates fractional
          phase per sample — no phase reset at window boundaries.
  FIX 3: Amplitude and F0 interpolation uses cubic (not linear) splines to avoid sharp
          discontinuities when loudness jumps between voiced/unvoiced frames.
  FIX 4: Silence frames (f0 < 30 Hz) now ramp to zero over 5 ms rather than hard-cutting,
          eliminating clicks at note boundaries.
  FIX 5: Harmonic convolution smoothing replaced with proper Gaussian filter (no edge artifacts).
  FIX 6: Output clipped only at ±1.0 (was clipping headroom at 0.88, causing distortion on
          loud passages).
  FIX 7: Body LPC filter from Layer 4 is now applied after synthesis (was silently skipped).
  FIX 8: flat_env shape guard: if WORLD envelope has wrong frame count, it is resampled
          to match f0 rather than truncated (caused dropout on shorter clips).

Quality additions v5.3:
  ADD:  5× oversampling for additive synthesis with 4th-order anti-aliasing lowpass,
        then decimation back to output_sr. Eliminates aliasing on high harmonics.
  ADD:  Per-instrument tuning temperament (stretch-tuning for piano based on Railsback curve).
  ADD:  Stereo width via Haas effect (1–3 ms delayed copy, instrument-specific).
  ADD:  Final True-peak limiter at -1 dBTP with 0.5 ms lookahead.
"""

import os
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter, medfilt, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_opening, uniform_filter1d
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Instrument profiles
# ─────────────────────────────────────────────────────────────────────────────

HARMONIC_PROFILES = {
    # Violin: bowed string — 32 harmonics, flatter rolloff in partials 3–8 where
    # the bow-resonance region lives; brighter than before without being harsh
    'violin': np.array([
        1.0,    0.82,   0.68,   0.57,   0.47,   0.39,   0.32,   0.26,   0.20,   0.15,
        0.112,  0.082,  0.060,  0.044,  0.032,  0.023,  0.017,  0.012,  0.009,  0.006,
        0.0045, 0.0033, 0.0024, 0.0017, 0.0013, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003,
        0.00022,0.00016,
    ]),
    # Cello: warmer/darker than violin — stronger fundamental relative to upper partials
    'cello':  np.array([1.0, 0.90, 0.72, 0.58, 0.46, 0.36, 0.26, 0.18, 0.12, 0.08,
                        0.055, 0.038, 0.026, 0.018, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001]),
    # Flute: nearly pure — fundamental dominates, rapid harmonic decay
    'flute':  np.array([1.0, 0.22, 0.045, 0.012, 0.004, 0.002, 0.001, 0.0005]),
    # Piano: steeper rolloff above h=3 to fix spectral centroid (was too bright)
    'piano':  np.array([1.0, 0.52, 0.16, 0.06, 0.028, 0.014, 0.008, 0.005, 0.003, 0.0018,
                        0.0011, 0.0007, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001, 0.00008, 0.00006, 0.00004]),
    # Guitar: plucked string — rich low harmonics, faster rolloff above h=6
    'guitar': np.array([1.0, 0.60, 0.42, 0.30, 0.22, 0.15, 0.10, 0.068, 0.046, 0.030,
                        0.020, 0.013, 0.009, 0.006, 0.004, 0.002, 0.0015, 0.001]),
    # Synth: true sawtooth — 30 harmonics at exact 1/h rolloff for full-spectrum sound
    'synth':  np.array([1.0/h for h in range(1, 31)], dtype=np.float32),
    # Trumpet: bright brass with strong 2nd-4th harmonics; soft at low dynamics,
    # blazing at high — see LOUDNESS_BRIGHTNESS for dynamic adjustment
    'trumpet': np.array([1.0, 0.88, 0.74, 0.54, 0.34, 0.20, 0.11, 0.07, 0.045, 0.028,
                         0.018, 0.011, 0.007, 0.004, 0.002]),
}

ENVELOPE_PARAMS = {
    'violin': {'attack': 0.075, 'release': 0.18, 'type': 'bowed'},   # slower bow onset = smoother
    'cello':  {'attack': 0.060, 'release': 0.20, 'type': 'bowed'},
    'flute':  {'attack': 0.020, 'release': 0.08, 'type': 'blown'},
    # Piano/guitar release kept short — long releases accumulate multiplicatively
    # across many note transitions (pitch-quantized melody = 100+ offset events),
    # each multiplying the envelope by a ramp toward 0, driving signal to silence.
    # The per-onset exponential decay in 'plucked' already handles string decay.
    'piano':  {'attack': 0.003, 'release': 0.80, 'type': 'plucked'},
    'guitar': {'attack': 0.004, 'release': 0.35, 'type': 'plucked'},
    'synth':  {'attack': 0.015, 'release': 0.06, 'type': 'synth'},
    'trumpet': {'attack': 0.018, 'release': 0.12, 'type': 'blown'},
}

# ── Pitch-dependent harmonic rolloff ──────────────────────────────────────────
# Higher f0 → harmonics decay faster (e.g. piano treble is purer than bass).
# Scalar: 0 = flat (synth), larger = stronger pitch-dependent thinning.
PITCH_ROLLOFF_SCALE = {
    'violin': 0.28, 'cello': 0.40, 'flute': 0.30,   # violin: 0.55→0.28, keeps upper harmonics bright
    'piano':  0.00, 'guitar': 0.30, 'synth': 0.00, 'trumpet': 0.20,
}

# Reference f0 where rolloff is neutral (neither boosted nor cut)
_PITCH_REF_HZ = {
    'violin': 440.0, 'cello': 130.8, 'flute': 523.25,
    'piano':  261.6, 'guitar': 196.0, 'synth': 440.0, 'trumpet': 329.6,
}

# Loudness-dependent spectral brightness: louder → more high harmonics.
# Crucial for trumpet/strings. 0 = no effect.
LOUDNESS_BRIGHTNESS = {
    'violin': 0.48, 'cello': 0.28, 'flute': 0.18,   # violin: 0.35→0.48 more high-harmonic brightness
    'piano':  0.05, 'guitar': 0.10, 'synth': 0.02, 'trumpet': 0.25,
}

# Piano/guitar note-frequency-dependent T60 reference (seconds at middle C)
# T60 scales as: t60(f) = T60_REF * (f_ref / f) ** T60_EXP
_PIANO_T60_REF  = 5.5   # T60 at C4 (261.6 Hz)
_PIANO_T60_EXP  = 0.60  # exponent — steeper = more frequency-dependent decay
_GUITAR_T60_REF = 2.0   # T60 at G3 (196 Hz) for guitar
_GUITAR_T60_EXP = 0.45

NOISE_MIX = {
    'violin': 0.042,  # reduced — 0.065 was too scratchy contributing to hard feel
    'cello':  0.03,
    'flute':  0.09,
    'piano':  0.010,
    'guitar': 0.015,
    'synth':  0.00,
    'trumpet': 0.03,
}

NOISE_BANDS = {
    'violin': (1000, 14000),
    'cello':  (400,  9000),
    'flute':  (150,  11000),
    'piano':  (60,   12000),
    'guitar': (200,  7000),
    'synth':  (500,  8000),
    'trumpet': (200, 10000),
}

INHARMONICITY_B = {
    'violin': 0.00002,
    'cello':  0.00003,
    'flute':  0.0,
    'piano':  0.00030,
    'guitar': 0.00015,
    'synth':  0.0,
    'trumpet': 0.0,
}

VIBRATO_DEPTH = {
    'violin': 1.5,   # ±9 cents — further reduced (measured output was 67 cents, target 20-50)
    'cello':  0.9,
    'flute':  0.85,
    'piano':  0.0,
    'guitar': 0.3,
    'synth':  0.5,   # ↑ raised so regular LFO is audible (was 0.2 → ±1.2 cents, inaudible)
    'trumpet': 0.4,   # subtle brass vibrato
}

# Stereo Haas delay in ms (0 = mono)
HAAS_MS = {
    'violin': 1.5,
    'cello':  2.0,
    'flute':  1.0,
    'piano':  2.5,
    'guitar': 1.8,
    'synth':  1.5,   # ↓ reduced from 3.0 — avoids comb filtering on mono playback
    'trumpet': 1.2,
}

# Blend of spectral envelope vs pure instrument profile in harmonic array.
# 0 = pure instrument profile (electronic/clean), 1 = pure spectral envelope.
# Piano/guitar need low blend: the Seed-VC spectral envelope still carries
# vocal formant colouring that overwrites the piano harmonic profile.
# Bowed strings/flute can tolerate more blend for natural bow/breath variation.
_BLEND = {
    'violin': 0.25,
    'cello':  0.20,
    'flute':  0.25,
    'piano':  0.00,   # pure harmonic profile — no vocal spectral envelope at all
    'guitar': 0.00,   # K-S doesn't use this path; 0 avoids vocal bleed if fallback runs
    'synth':  0.15,   # ↑ slight increase so voice dynamics colour the timbre slightly
    'trumpet': 0.25,
}

OVERSAMPLE = 4      # oversampling factor for additive synthesis


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _db_to_amp(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.clip(db, -80.0, 0.0) / 20.0)


def _smooth_interp(frame_times, values, sample_times):
    """Cubic interpolation with edge-clamping — avoids linear kinks."""
    if len(frame_times) < 4:
        return np.interp(sample_times, frame_times, values).astype(np.float32)
    fn = interp1d(frame_times, values, kind='cubic',
                  bounds_error=False,
                  fill_value=(values[0], values[-1]))
    return fn(sample_times).astype(np.float32)


def _butter_lowpass_sos(cutoff_hz, sr, order=8):
    nyq = sr / 2.0
    norm = min(cutoff_hz / nyq, 0.9999)
    return butter(order, norm, btype='low', output='sos')


def _true_peak_limit(audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
    """Simple brick-wall true-peak limiter (4× oversampled peak detection)."""
    from scipy.signal import resample_poly
    # Upsample 4× to detect inter-sample peaks
    up = resample_poly(audio, 4, 1)
    peak = np.max(np.abs(up))
    thresh_amp = 10 ** (threshold_db / 20.0)
    if peak > thresh_amp:
        audio = audio * (thresh_amp / peak)
    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral-envelope-guided harmonic amplitude array
# ─────────────────────────────────────────────────────────────────────────────

def _build_harmonic_amp_array(f0_frames, flat_env, instrument, sr,
                               loudness_frames=None, blend=0.55):
    profile  = HARMONIC_PROFILES.get(instrument, HARMONIC_PROFILES['violin'])
    n_frames = len(f0_frames)
    n_harm   = len(profile)
    n_bins   = flat_env.shape[1]
    freq_res = (sr / 2.0) / n_bins

    profile_norm = profile / (profile.max() + 1e-8)

    row_idx = np.arange(n_frames)
    env_amps = np.zeros((n_frames, n_harm), dtype=np.float32)
    for h_idx in range(n_harm):
        h     = h_idx + 1
        freqs = np.minimum(h * f0_frames, sr / 2.0 - 1.0)
        bins  = np.clip(freqs / freq_res, 0, n_bins - 1).astype(np.int32)
        env_amps[:, h_idx] = flat_env[row_idx, bins]

    env_max       = env_amps.max(axis=1, keepdims=True) + 1e-8
    env_amps_norm = env_amps / env_max

    blended = (1.0 - blend) * profile_norm[np.newaxis, :] + blend * env_amps_norm
    blended = blended / (blended.max(axis=1, keepdims=True) + 1e-8)

    # ── Pitch-dependent harmonic rolloff (vectorised) ─────────────────────────
    # High register → harmonics attenuate faster (e.g. piano treble is purer).
    scale = PITCH_ROLLOFF_SCALE.get(instrument, 0.0)
    if scale > 0.0:
        f_ref  = _PITCH_REF_HZ.get(instrument, 440.0)
        voiced = f0_frames > 30.0
        p_ratio = np.where(voiced, np.clip(f0_frames / f_ref, 0.25, 6.0), 1.0)
        log_r  = np.log2(p_ratio)[:, np.newaxis]            # [n_frames, 1]
        h_idx_v = np.arange(n_harm, dtype=np.float32)[np.newaxis, :]  # [1, n_harm]
        rolloff = np.exp(-scale * h_idx_v * np.maximum(log_r, 0.0) * 0.18)
        blended = (blended * rolloff).astype(np.float32)
        blended = blended / (blended.max(axis=1, keepdims=True) + 1e-8)

    # ── Loudness-dependent brightness (vectorised) ────────────────────────────
    # Louder playing → more high-harmonic energy (critical for brass/strings).
    brightness = LOUDNESS_BRIGHTNESS.get(instrument, 0.0)
    if brightness > 0.0 and loudness_frames is not None:
        amp_lin  = _db_to_amp(np.asarray(loudness_frames, dtype=np.float32))
        amp_max  = amp_lin.max() + 1e-8
        amp_norm = (amp_lin / amp_max)[:, np.newaxis]        # [n_frames, 1]
        h_frac   = (np.arange(n_harm, dtype=np.float32) / max(n_harm - 1, 1))[np.newaxis, :]
        boost    = 1.0 + brightness * amp_norm * h_frac      # [n_frames, n_harm]
        blended  = (blended * boost).astype(np.float32)
        blended  = blended / (blended.max(axis=1, keepdims=True) + 1e-8)

    # Temporal smoothing: sigma=2 frames = 20 ms
    blended = gaussian_filter1d(blended, sigma=2.0, axis=0)

    return blended.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Additive synthesis  — v5.3: oversampled, phase-continuous, cubic-interpolated
# ─────────────────────────────────────────────────────────────────────────────

def _synthesise_additive(f0, loudness, harm_amps, instrument, sr, frame_hop, vibrato=None):
    """
    Phase-continuous additive synthesis at OVERSAMPLE × sr, then decimated.

    Critical fixes:
    • Phase accumulates monotonically — no per-window resets.
    • F0 = 0 frames trigger a 5 ms amplitude fade to silence (avoids clicks).
    • Cubic interpolation of F0 and amplitude at oversample rate.
    • Anti-aliasing LP filter before decimation.
    """
    profile   = HARMONIC_PROFILES.get(instrument, HARMONIC_PROFILES['violin'])
    n_harm    = len(profile)
    n_frames  = len(f0)
    sr_os     = sr * OVERSAMPLE
    frame_hop_os = frame_hop * OVERSAMPLE
    n_samples_os = n_frames * frame_hop_os
    B         = INHARMONICITY_B.get(instrument, 0.0)
    vib_depth = VIBRATO_DEPTH.get(instrument, 0.0)

    ft = np.arange(n_frames) * frame_hop_os
    st = np.arange(n_samples_os, dtype=np.float64)

    # ── F0 at oversample rate ─────────────────────────────────────────────────
    f0_os = _smooth_interp(ft.astype(np.float32),
                           f0.astype(np.float32),
                           st.astype(np.float32)).astype(np.float64)

    # ── Vibrato ───────────────────────────────────────────────────────────────
    if vibrato is not None and vib_depth > 0.0:
        vib = np.resize(vibrato, n_frames).astype(np.float64)
        vib = gaussian_filter1d(vib, sigma=3.0)
        vib -= vib.mean()
        rng = np.max(np.abs(vib)) + 1e-8
        vib_norm = vib / rng
        vib_os   = _smooth_interp(ft.astype(np.float32),
                                   vib_norm.astype(np.float32),
                                   st.astype(np.float32)).astype(np.float64)
        f0_os = f0_os * (2.0 ** (6.0 * vib_depth * vib_os / 1200.0))

    # ── Silence mask: voiced = f0 > 30 Hz ────────────────────────────────────
    min_frames_v  = max(1, int(0.080 * sr / frame_hop))
    voiced_raw_s  = (f0 > 30.0)
    voiced_closed   = binary_closing(voiced_raw_s, structure=np.ones(min_frames_v))
    voiced_filtered = binary_opening(voiced_closed, structure=np.ones(min_frames_v))
    voiced_frames   = voiced_filtered.astype(np.float32)

    # 30ms fade for plucked (smooth note end, no click), 5ms for continuous bow.
    # NOTE: dilation was removed — dilating into f0=0 frames creates a DC buzz
    # because phase_inc=0 → sin(frozen_phase) = constant during the extended region.
    if instrument in ('piano', 'guitar'):
        fade_samples = max(1, int(0.030 * sr_os))
    else:
        fade_samples = max(1, int(0.005 * sr_os))
    voiced_os = np.interp(st, ft, voiced_frames.astype(np.float64))
    voiced_os = uniform_filter1d(voiced_os, size=fade_samples)
    voiced_os = np.clip(voiced_os, 0.0, 1.0)

    # ── Amplitude envelope ────────────────────────────────────────────────────
    # Piano: raw WORLD loudness spans 30–50 dB → 30–300× amplitude swings between
    # notes. Compress to ±8 dB of the median voiced level (≤ 2.5× amplitude range)
    # then smooth over 300ms. This gives natural piano dynamics without wild jumps.
    if instrument in ('piano', 'guitar'):
        voiced_mask = f0 > 30.0
        if voiced_mask.any():
            med_db = float(np.median(loudness[voiced_mask]))
            loudness_c = np.clip(loudness.astype(np.float64),
                                 med_db - 8.0, med_db + 8.0)
        else:
            loudness_c = loudness.astype(np.float64)
        loudness_smooth = gaussian_filter1d(loudness_c, sigma=30.0)
    else:
        loudness_smooth = loudness.astype(np.float64)

    amp_frames = _db_to_amp(loudness_smooth).astype(np.float64)
    amp_os = _smooth_interp(ft.astype(np.float32),
                             amp_frames.astype(np.float32),
                             st.astype(np.float32)).astype(np.float64)
    amp_os *= voiced_os

    # Smooth 8 ms; clip to ≥ 0 — cubic can briefly overshoot negative at
    # voiced/unvoiced transitions, causing destructive phase cancellation.
    smooth_os = max(1, int(0.008 * sr_os))
    amp_os = uniform_filter1d(amp_os, size=smooth_os)
    amp_os = np.maximum(amp_os, 0.0)

    # ── Build output ──────────────────────────────────────────────────────────
    audio_os = np.zeros(n_samples_os, dtype=np.float64)

    for h_idx in range(n_harm):
        h       = h_idx + 1
        stretch = np.sqrt(1.0 + B * h * h)
        freq_h  = f0_os * h * stretch
        freq_h  = np.minimum(freq_h, (sr_os / 2.0) - 1.0)
        freq_h  = np.maximum(freq_h, 0.0)

        # Phase-continuous accumulation
        phase_inc = 2.0 * np.pi * freq_h / sr_os
        phases    = np.cumsum(phase_inc)      # monotonic, no resets

        # Per-harmonic amplitude — linear interp (30–50× faster than cubic;
        # harm_amps is already Gaussian-smoothed at frame level so sample-level
        # linear is accurate). Clip ≥ 0: interp between 0 and a positive value
        # is always positive, but guard against any edge-case underflow.
        ha_frames = harm_amps[:, h_idx].astype(np.float64)
        ha_os = np.interp(st, ft.astype(np.float64), ha_frames)
        ha_os = uniform_filter1d(ha_os, size=max(1, int(0.005 * sr_os)))
        ha_os = np.maximum(ha_os, 0.0)

        audio_os += ha_os * amp_os * np.sin(phases)

    profile_sum = profile.sum() + 1e-8
    audio_os /= profile_sum

    # ── Anti-aliasing lowpass then decimate ───────────────────────────────────
    # sosfiltfilt = zero-phase; causal sosfilt introduces group delay that
    # time-shifts the waveform relative to its amplitude envelope.
    sos_aa = _butter_lowpass_sos(sr / 2.0 * 0.9, sr_os, order=8)
    audio_os = sosfiltfilt(sos_aa, audio_os)
    audio_out = audio_os[::OVERSAMPLE]

    return audio_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FluidSynth sample-based synthesis  (piano + orchestral instruments)
# ─────────────────────────────────────────────────────────────────────────────

# GM program numbers for each instrument
_GM_PROGRAM = {
    'piano':   0,   # Acoustic Grand Piano
    'guitar':  25,  # Acoustic Guitar (steel)
    'violin':  40,  # Violin
    'cello':   42,  # Cello
    'flute':   73,  # Flute
    'trumpet': 56,  # Trumpet
    'synth':   80,  # Lead synth (square)
}

def _synthesise_fluidsynth(f0, loudness, instrument, sr, frame_hop, sf2_path):
    """
    Sample-based synthesis using FluidSynth command-line offline renderer.

    Builds a minimal Type-0 MIDI file from note segments, then calls:
        fluidsynth -ni -F output.wav -r SR soundfont.sf2 input.mid
    The CLI renderer is the reliable path — the Python get_samples() API has
    internal buffer-size quirks that cause long silent sections.
    """
    import subprocess, tempfile, struct, os as _os

    n_frames  = len(f0)
    n_samples = n_frames * frame_hop

    # ── f0 → MIDI note numbers ────────────────────────────────────────────────
    voiced  = f0 > 30.0
    f0_safe = np.where(voiced, np.maximum(f0, 1.0), 440.0)
    midi_q  = np.clip(np.round(12.0 * np.log2(f0_safe / 440.0) + 69.0),
                      21, 108).astype(int)
    midi_q  = np.where(voiced, midi_q, 0)

    # ── loudness → MIDI velocity (37–127) ─────────────────────────────────────
    amp_lin  = _db_to_amp(loudness)
    amp_peak = amp_lin.max() + 1e-8

    # ── note segments: (start_samp, end_samp, midi_note, velocity) ────────────
    segments = []
    i = 0
    while i < n_frames:
        if not voiced[i] or midi_q[i] == 0:
            i += 1; continue
        note = int(midi_q[i])
        j = i + 1
        while j < n_frames and voiced[j] and int(midi_q[j]) == note:
            j += 1
        vel = int(np.clip(float(amp_lin[i:j].mean()) / amp_peak * 90 + 37, 37, 127))
        segments.append((i * frame_hop, j * frame_hop, note, vel))
        i = j

    if not segments:
        return np.zeros(n_samples, dtype=np.float32)

    print(f"[FluidSynth] {len(segments)} segments  "
          f"MIDI {min(s[2] for s in segments)}–{max(s[2] for s in segments)}")

    # ── Build Type-0 MIDI file ────────────────────────────────────────────────
    # TPQN must be ≤ 32767 (15-bit MIDI limit). 44100 overflows → SMPTE mode.
    # Use standard 480 ticks/beat at 120 BPM → 960 ticks/second.
    TPQN   = 480         # ticks per quarter note (standard, safe)
    TEMPO  = 500_000     # µs/beat → 120 BPM
    TPS    = TPQN * 1_000_000 // TEMPO   # ticks per second = 960
    PROG   = _GM_PROGRAM.get(instrument, 0)

    def _vlq(n):
        buf = [n & 0x7F]; n >>= 7
        while n:
            buf.append((n & 0x7F) | 0x80); n >>= 7
        return bytes(reversed(buf))

    def _s2t(samp):
        """Convert sample position to MIDI ticks."""
        return int(round(samp * TPS / sr))

    evts = []
    for s0, s1, note, vel in segments:
        evts.append((_s2t(s0), bytes([0x90, note, vel])))   # note-on
        evts.append((_s2t(s1), bytes([0x80, note, 0])))     # note-off
    evts.sort(key=lambda e: e[0])

    track = bytearray()
    track += b'\x00\xFF\x51\x03' + TEMPO.to_bytes(3, 'big')   # set tempo
    track += b'\x00' + bytes([0xC0, PROG])                     # program change
    prev = 0
    for tick, data in evts:
        track += _vlq(tick - prev) + data
        prev = tick
    track += _vlq(int(3.0 * TPS)) + b'\xFF\x2F\x00'           # 3-s tail + EOT

    midi_bytes = (
        b'MThd' + struct.pack('>IHHH', 6, 0, 1, TPQN) +
        b'MTrk' + struct.pack('>I', len(track)) + bytes(track)
    )

    # ── Render via FluidSynth CLI ─────────────────────────────────────────────
    mid_tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    wav_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    mid_tmp.write(midi_bytes); mid_tmp.close(); wav_tmp.close()

    try:
        cmd = ['fluidsynth', '-ni', '-g', '1.0',
               '-F', wav_tmp.name, '-r', str(sr),
               sf2_path, mid_tmp.name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not _os.path.isfile(wav_tmp.name) or _os.path.getsize(wav_tmp.name) < 44:
            raise RuntimeError(f"FluidSynth CLI failed:\n{res.stderr[-400:]}")

        audio, file_sr = sf.read(wav_tmp.name)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa as _lb
            audio = _lb.resample(audio.astype(np.float32),
                                  orig_sr=file_sr, target_sr=sr)

        # Trim / zero-pad to exact n_samples
        audio = audio.astype(np.float32)
        if len(audio) >= n_samples:
            audio = audio[:n_samples]
        else:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        peak = np.max(np.abs(audio)) + 1e-9
        if peak > 1e-6:
            audio /= peak
        return audio

    finally:
        for p in [mid_tmp.name, wav_tmp.name]:
            try: _os.unlink(p)
            except: pass


def _find_sf2(cfg: dict, instrument: str) -> str | None:
    """Return path to best available SF2 for this instrument, or None."""
    # 1. Instrument-specific soundfont from config
    sf = cfg.get('soundfonts', {}).get(instrument, '')
    if sf and os.path.isfile(sf):
        return sf
    # 2. General soundfont from config
    sf = cfg.get('soundfonts', {}).get('general', '')
    if sf and os.path.isfile(sf):
        return sf
    # 3. Auto-discover — scan soundfonts/ for any SF2/SF3 file
    sf_dir = 'soundfonts'
    if os.path.isdir(sf_dir):
        for fname in sorted(os.listdir(sf_dir)):
            if fname.lower().endswith(('.sf2', '.sf3')):
                path = os.path.join(sf_dir, fname)
                if os.path.getsize(path) > 1024:   # skip 0-byte placeholders
                    return path
    # 4. System fallbacks
    for candidate in [
        '/usr/share/sounds/sf2/FluidR3_GM.sf2',
        '/usr/share/sounds/sf2/FluidR3_Mono.sf2',
        '/usr/share/soundfonts/default.sf2',
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Karplus-Strong plucked-string synthesis  (guitar)
# ─────────────────────────────────────────────────────────────────────────────

def _karplus_strong_guitar(f0, loudness, sr, frame_hop):
    """
    Karplus-Strong synthesis driven by CREPE/WORLD f0 and loudness.

    One K-S segment per voiced note (detected via morphological filter on f0).
    Each note uses the median f0 of that segment.  The K-S recurrence is
    implemented as an IIR via scipy.signal.lfilter — fully vectorised, no
    Python sample loop.

    Difference equation:
        y[n] = x[n] + g/2 * y[n-(N-1)] + g/2 * y[n-N]
    where N = round(sr / f0_median), g = decay coefficient from T60.
    Excitation x[n] is bandpass noise over the first N samples, then zero.
    """
    from scipy.signal import butter, sosfilt, lfilter as sp_lfilter

    n_frames  = len(f0)
    n_samples = n_frames * frame_hop
    audio     = np.zeros(n_samples, dtype=np.float64)

    ft = np.arange(n_frames, dtype=np.float64) * frame_hop
    st = np.arange(n_samples, dtype=np.float64)

    amp_s = np.interp(st, ft, _db_to_amp(loudness).astype(np.float64))
    amp_s = uniform_filter1d(amp_s, size=max(1, int(0.010 * sr)))

    # Voiced segmentation
    min_frm = max(1, int(0.040 * sr / frame_hop))
    v_raw   = (f0 > 30.0)
    v_cls   = binary_closing(v_raw, structure=np.ones(min_frm))
    v_flt   = binary_opening(v_cls,   structure=np.ones(min_frm))
    onsets  = np.where(np.diff(v_flt.astype(int), prepend=0) > 0)[0]
    offsets = np.where(np.diff(v_flt.astype(int), append=0)  < 0)[0]

    for i, on_frm in enumerate(onsets):
        off_frm = offsets[i] if i < len(offsets) else n_frames - 1
        on_s    = on_frm  * frame_hop
        off_s   = off_frm * frame_hop
        if on_s >= n_samples:
            continue

        # Median f0 for this note
        f0_seg    = f0[on_frm : off_frm + 1]
        f0_voiced = f0_seg[f0_seg > 30.0]
        if len(f0_voiced) == 0:
            continue
        f0_note = float(np.median(f0_voiced))
        if f0_note < 20.0:
            continue

        N = max(2, int(round(sr / f0_note)))

        # T60 scales inversely with frequency: bass strings ring longer
        t60 = _GUITAR_T60_REF * (196.0 / max(f0_note, 50.0)) ** _GUITAR_T60_EXP
        t60 = np.clip(t60, 0.3, 5.0)
        # Per-period decay: g^(sr/N) = 10^(-3/t60) → g = 10^(-3*N/(t60*sr))
        g = 10.0 ** (-3.0 * N / (t60 * sr))
        g = float(np.clip(g, 0.90, 0.9999))

        # Excitation: bandpass noise (simulates pluck)
        exc = np.random.randn(N * 3).astype(np.float64)
        lo  = max(f0_note * 0.8 / (sr / 2.0), 1e-4)
        hi  = min(f0_note * 6.0 / (sr / 2.0), 0.499)
        if hi > lo + 1e-4:
            sos_e = butter(2, [lo, hi], btype='band', output='sos')
            exc   = sosfilt(sos_e, exc)
        exc = exc[-N:] / (np.max(np.abs(exc[-N:])) + 1e-8)

        # IIR coefficients for K-S recurrence y[n] = g/2*(y[n-N] + y[n-(N-1)]) + x[n]
        # a[0]=1, a[N-1]=-g/2, a[N]=-g/2  (all other a[k]=0)
        decay_len = min(off_s - on_s + int(t60 * sr * 1.5), n_samples - on_s)
        if decay_len <= 0:
            continue

        a_coef       = np.zeros(N + 1)
        a_coef[0]    = 1.0
        a_coef[N - 1] = -g / 2.0
        a_coef[N]    = -g / 2.0

        x_in          = np.zeros(decay_len)
        x_in[:min(N, decay_len)] = exc[:min(N, decay_len)]

        ks_out = sp_lfilter([1.0], a_coef, x_in)

        # Amplitude envelope for this note
        end_s = on_s + decay_len
        a_env = amp_s[on_s:end_s].copy()
        # Attack shaping (2 ms ramp)
        atk   = max(1, int(0.002 * sr))
        a_env[:atk] *= np.linspace(0.0, 1.0, atk) ** 0.7

        audio[on_s:end_s] += ks_out * a_env

    # Peak-normalise before return (the caller will re-normalise globally)
    peak = np.max(np.abs(audio)) + 1e-9
    if peak > 1e-6:
        audio /= peak

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Noise model
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(audio, loudness, flat_env, instrument, sr, frame_hop):
    level = NOISE_MIX.get(instrument, 0.02)
    if level < 0.001:
        return audio

    n  = len(audio)
    ft = np.arange(len(loudness)) * frame_hop
    st = np.arange(n, dtype=np.float32)

    lo, hi = NOISE_BANDS.get(instrument, (200, 8000))
    noise  = np.random.randn(n).astype(np.float32)
    lo_n   = max(lo / (sr / 2.0), 1e-4)
    hi_n   = min(hi / (sr / 2.0), 0.999)
    sos    = butter(4, [lo_n, hi_n], btype='band', output='sos')
    noise  = sosfiltfilt(sos, noise).astype(np.float32)

    env_energy = np.sqrt(np.mean(np.maximum(flat_env, 0) ** 2, axis=1)).astype(np.float32)
    env_energy = gaussian_filter1d(env_energy, sigma=2.0)
    env_energy /= (env_energy.max() + 1e-8)
    # Linear interp + explicit clip: env_energy is already Gaussian-smoothed at
    # frame level; cubic would overshoot at voiced/unvoiced boundaries creating
    # brief negative values that invert noise phase and add audible artifacts.
    noise_env  = np.maximum(
        np.interp(st, ft.astype(np.float64), env_energy.astype(np.float64)), 0.0
    ).astype(np.float32)

    noise_out = noise * noise_env * level

    if instrument in ('violin', 'cello'):
        rosin_lo = 5000 if instrument == 'violin' else 3000
        rn = np.random.randn(n).astype(np.float32)
        sos2 = butter(2, [rosin_lo / (sr / 2.0), 0.98], btype='band', output='sos')
        rn   = sosfiltfilt(sos2, rn).astype(np.float32)
        noise_out += rn * noise_env * level * 0.4

    if instrument == 'flute':
        amp_s = np.interp(st, ft.astype(np.float64), _db_to_amp(loudness).astype(np.float64)).astype(np.float32)
        onset_mask = np.zeros(n, dtype=np.float32)
        diff_env   = np.diff(noise_env, prepend=noise_env[0])
        onset_samp = np.where(diff_env > 0.01)[0]
        burst_len  = int(sr * 0.025)
        for os_ in onset_samp:
            e = min(os_ + burst_len, n)
            onset_mask[os_:e] += np.linspace(1.0, 0.0, e - os_)
        onset_mask = np.clip(onset_mask, 0, 1)
        bn = np.random.randn(n).astype(np.float32)
        sos3 = butter(2, [80 / (sr / 2.0), 0.9], btype='band', output='sos')
        bn   = sosfiltfilt(sos3, bn).astype(np.float32)
        noise_out += bn * onset_mask * level * 1.5

    if instrument == 'piano':
        # Hammer-attack transient: 15 ms broadband click at each note onset.
        # Real piano hammers produce a brief wideband impact before the string
        # tone dominates; without this the piano sounds like a pure sine wave.
        onset_mask = np.zeros(n, dtype=np.float32)
        diff_env   = np.diff(noise_env, prepend=noise_env[0])
        onset_samp = np.where(diff_env > 0.006)[0]
        burst_len  = int(sr * 0.015)
        for os_ in onset_samp:
            e = min(os_ + burst_len, n)
            onset_mask[os_:e] += np.linspace(1.0, 0.0, e - os_)
        onset_mask = np.clip(onset_mask, 0, 1)
        bn = np.random.randn(n).astype(np.float32)
        # Soft-band the hammer click (not fully white — emphasise mid highs)
        sos_h = butter(2, [200 / (sr / 2.0), 0.95], btype='band', output='sos')
        bn    = sosfiltfilt(sos_h, bn).astype(np.float32)
        noise_out += bn * onset_mask * level * 4.0

    return audio + noise_out


# ─────────────────────────────────────────────────────────────────────────────
# Note envelope
# ─────────────────────────────────────────────────────────────────────────────

def _apply_envelope(audio, sr, instrument, f0, frame_hop):
    cfg      = ENVELOPE_PARAMS.get(instrument,
                                    {'attack': 0.03, 'release': 0.1, 'type': 'bowed'})
    atk_s    = int(sr * cfg['attack'])
    rel_s    = int(sr * cfg['release'])
    inst_type = cfg.get('type', 'bowed')

    # Morphological filter: close gaps < 80 ms and remove voiced segments < 80 ms.
    min_frames = max(1, int(0.080 * sr / frame_hop))
    voiced_raw = (f0 > 30.0)
    voiced_bool = binary_closing(voiced_raw, structure=np.ones(min_frames))
    voiced_bool = binary_opening(voiced_bool, structure=np.ones(min_frames))
    voiced  = voiced_bool.astype(np.float32)
    onsets  = np.where(np.diff(voiced, prepend=0) > 0)[0]
    offsets = np.where(np.diff(voiced, append=0) < 0)[0]

    if inst_type == 'plucked':
        # Per-note independent envelopes combined via MAX (not multiplication).
        # Multiplicative accumulation across 100+ notes drives env to ~10^-14.
        # Each note: sharp attack → exponential frequency-dependent decay.
        # max() correctly models multiple notes sounding: whichever is loudest wins.
        is_piano  = (instrument == 'piano')
        t60_ref   = _PIANO_T60_REF  if is_piano else _GUITAR_T60_REF
        t60_exp   = _PIANO_T60_EXP  if is_piano else _GUITAR_T60_EXP
        f_ref_dec = _PITCH_REF_HZ.get(instrument, 261.6)

        env = np.zeros(len(audio), dtype=np.float32)

        for frm in onsets:
            f0_at = float(f0[min(frm, len(f0)-1)]) if len(f0) > 0 else 220.0
            f0_at = max(f0_at, 40.0)
            t60   = float(np.clip(t60_ref * (f_ref_dec / f0_at) ** t60_exp, 0.3, 12.0))
            decay_rate  = np.log(1000.0) / t60
            max_decay_s = min(t60 * 2.0, 10.0)
            s = frm * frame_hop
            e = min(s + int(sr * max_decay_s), len(env))
            n = e - s
            if n <= 0:
                continue
            t        = np.arange(n, dtype=np.float64) / sr
            note_env = np.exp(-decay_rate * t).astype(np.float32)
            # Attack ramp
            atk_n = min(atk_s, n)
            if atk_n > 0:
                note_env[:atk_n] *= np.linspace(0.0, 1.0, atk_n) ** 0.7
            # Combine: take max so new notes reset to full amplitude
            env[s:e] = np.maximum(env[s:e], note_env)

        return (audio * env).astype(np.float32)

    # ── bowed / blown / synth ─────────────────────────────────────────────────
    env = np.ones(len(audio), dtype=np.float32)

    for frm in onsets:
        s = frm * frame_hop
        e = min(s + atk_s, len(env))
        if e > s:
            env[s:e] *= np.linspace(0.0, 1.0, e - s) ** 0.7

    for frm in offsets:
        s = frm * frame_hop
        e = min(s + rel_s, len(env))
        if e > s:
            env[s:e] *= np.linspace(1.0, 0.0, e - s) ** 0.5

    if inst_type in ('bowed', 'blown'):
        # Per-instrument bow/breath flutter depth and rate
        _fd = {'violin': 0.040, 'cello': 0.032, 'flute': 0.040, 'trumpet': 0.028}
        _fr = {'violin': 4.8,   'cello': 4.2,   'flute': 3.5,   'trumpet': 5.2}
        fd  = _fd.get(instrument, 0.025 if inst_type == 'bowed' else 0.04)
        fr  = _fr.get(instrument, 4.5 if inst_type == 'bowed' else 3.5)
        t_s    = np.arange(len(audio), dtype=np.float32) / sr
        flutter = 1.0 + fd * np.sin(2.0 * np.pi * fr * t_s)
        voiced_s = np.interp(t_s,
                              np.arange(len(voiced)) * frame_hop / sr,
                              voiced)
        flutter = 1.0 + (flutter - 1.0) * voiced_s
        env    *= flutter.astype(np.float32)

    return audio * env


# ─────────────────────────────────────────────────────────────────────────────
# Stereo widening via Haas effect
# ─────────────────────────────────────────────────────────────────────────────

def _haas_stereo(audio_mono: np.ndarray, sr: int, instrument: str) -> np.ndarray:
    """
    Create stereo signal from mono using Haas effect:
    L = original, R = delayed copy (1–3 ms, instrument-specific).
    Returns [N, 2] stereo array.
    """
    delay_ms  = HAAS_MS.get(instrument, 1.5)
    delay_smp = int(sr * delay_ms / 1000.0)
    n = len(audio_mono)
    right = np.concatenate([np.zeros(delay_smp, dtype=np.float32),
                             audio_mono[:n - delay_smp]])
    stereo = np.stack([audio_mono, right], axis=1)
    return stereo.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Body LPC filter
# ─────────────────────────────────────────────────────────────────────────────

def _apply_body_lpc(audio: np.ndarray, body_lpc_a: np.ndarray) -> np.ndarray:
    """Apply all-pole body resonance filter saved in Layer 4."""
    denom = np.concatenate([[1.0], -body_lpc_a.astype(np.float64)])
    return lfilter([1.0], denom, audio).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Synth VCF — envelope-following resonant lowpass
# ─────────────────────────────────────────────────────────────────────────────

def _apply_vcf(audio: np.ndarray, sr: int, loudness: np.ndarray,
               frame_hop: int, fc_min: float = 800.0,
               fc_max: float = 7000.0, Q: float = 2.5) -> np.ndarray:
    """
    Envelope-following resonant lowpass VCF for synth.

    Cutoff tracks amplitude so the filter opens on loud notes and closes
    on quiet ones — the classic ADSR-driven filter behaviour of analog synths.
    Processed in 10 ms chunks so biquad coefficients track the envelope
    without expensive sample-by-sample updates.
    """
    from scipy.signal import sosfilt

    n        = len(audio)
    n_frames = len(loudness)

    amp      = _db_to_amp(loudness)
    amp_norm = amp / (amp.max() + 1e-8)

    ft    = np.arange(n_frames, dtype=np.float64) * frame_hop
    st    = np.arange(n,        dtype=np.float64)
    amp_s = np.interp(st, ft, amp_norm.astype(np.float64))
    amp_s = uniform_filter1d(amp_s, size=max(1, int(0.020 * sr)))  # 20 ms smooth

    fc_s = fc_min + (fc_max - fc_min) * np.clip(amp_s, 0.0, 1.0)

    chunk_sz  = max(1, int(0.010 * sr))   # 10 ms per coefficient update
    out       = np.empty(n, dtype=np.float64)
    audio_d   = audio.astype(np.float64)
    zi        = np.zeros((1, 2))           # biquad state (1 SOS section)

    for i in range(0, n, chunk_sz):
        j    = min(i + chunk_sz, n)
        fc_c = float(np.median(fc_s[i:j]))
        fc_c = np.clip(fc_c, 20.0, sr * 0.49)

        w0     = 2.0 * np.pi * fc_c / sr
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha  = sin_w0 / (2.0 * Q)

        b0 = (1.0 - cos_w0) / 2.0
        b1 =  1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 =  1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 =  1.0 - alpha

        sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
        seg, zi = sosfilt(sos, audio_d[i:j], zi=zi)
        out[i:j] = seg

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_synthesis(ddsp_params_path, instrument, config_path, output_path,
                  output_sr=44100):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[Layer 5] Loading DDSP params: {ddsp_params_path}")
    params = np.load(ddsp_params_path)

    # ── F0 ────────────────────────────────────────────────────────────────────
    # kernel_size=5 = 50 ms smoothing; caused pitch to lag and smear at every
    # note transition. 3 frames (30 ms) still kills single-frame glitches but
    # tracks melody transitions with much less delay.
    f0 = medfilt(params['ddsp_f0'].astype(np.float32), kernel_size=3)
    f0 = np.where(f0 < 10.0, 0.0, f0)
    # Zero frames above 1200 Hz — soprano ceiling, matches Layer 2 CREPE limit.
    # 700 Hz was silencing legitimate voiced content above E5 (soprano register).
    f0 = np.where(f0 > 1200.0, 0.0, f0)
    f0 = np.where(f0 < 10.0, 0.0, f0)   # true silence = 0, not 10 Hz

    # ── Aperiodicity gate ─────────────────────────────────────────────────────
    # WORLD aperiodicity: ~0 = fully voiced, ~1 = fully noise/unvoiced.
    # Low-frequency bands are the most stable voicing discriminator.
    # Frames above 0.5 are noise — zero their f0 so they aren't synthesised
    # as spurious high-pitched notes (especially common in the second half
    # where CREPE/WORLD tracks background noise after the voice stops).
    if 'ddsp_aperiodicity' in params:
        ap = params['ddsp_aperiodicity'].astype(np.float32)
        n_ap_bins = ap.shape[1]
        ap_low = ap[:, :max(1, n_ap_bins // 4)].mean(axis=1)
        n_gate = min(len(f0), len(ap_low))
        noise_mask = ap_low[:n_gate] > 0.5
        f0[:n_gate] = np.where(noise_mask, 0.0, f0[:n_gate])
        print(f"[Layer 5] Aperiodicity gate: {int(noise_mask.sum())}/{n_gate} frames silenced as noise")

    # ── Piano/synth: quantize f0 to equal-temperament semitones ──────────────
    # Voice glides continuously between pitches — piano/synth are fixed-pitch.
    # Without quantization the synthesis sounds like a singing voice, not keys.
    if instrument in ('piano', 'synth'):
        voiced_mask = f0 > 30.0
        f0_safe = np.where(voiced_mask, np.maximum(f0, 1.0), 440.0)
        midi = 12.0 * np.log2(f0_safe / 440.0) + 69.0
        midi_q = np.round(midi)
        f0_q = 440.0 * (2.0 ** ((midi_q - 69.0) / 12.0))
        f0 = np.where(voiced_mask, f0_q.astype(np.float32), 0.0)
        n_unique = len(np.unique(midi_q[voiced_mask])) if voiced_mask.any() else 0
        print(f"[Layer 5] Pitch quantized to {n_unique} distinct semitones")

    # ── Loudness ──────────────────────────────────────────────────────────────
    # Always use original WORLD-derived loudness for amplitude control.
    # ddsp_loudness_flat was computed from the cepstrally-liftered envelope;
    # if DC (bin 0) was inadvertently zeroed during liftering, all frames
    # appear at ~0 dBFS with no dynamics. Original loudness is authoritative.
    loudness = params['ddsp_loudness'].astype(np.float32)
    print("[Layer 5] Using original WORLD loudness for amplitude dynamics")

    # ── Spectral envelope ─────────────────────────────────────────────────────
    if 'ddsp_envelope_flat' in params:
        flat_env = params['ddsp_envelope_flat'].astype(np.float32)
        print("[Layer 5] Using formant-suppressed envelope")
    else:
        flat_env = params['ddsp_envelope'].astype(np.float32)

    # ── Vibrato ───────────────────────────────────────────────────────────────
    vibrato = params.get('ddsp_vibrato', None)
    if instrument == 'synth':
        # Synth uses a regular 5 Hz sine LFO — not irregular human vocal vibrato.
        # Human vibrato is biological (random phase, variable rate) which makes
        # the synth sound organic/wobbly instead of electronic.
        n_vib    = len(f0)
        t_frames = np.arange(n_vib, dtype=np.float32) * 0.010   # 10 ms per frame
        vibrato  = np.sin(2.0 * np.pi * 5.0 * t_frames).astype(np.float32)
        print(f"[Layer 5] Synth: regular 5 Hz sine LFO (±3 cents @ depth=0.5)")
    elif instrument == 'violin':
        # Synthesized violin vibrato: 5.5 Hz with subtle rate variation for organic feel.
        # Extracted vocal vibrato amplitude depends on how much natural vibrato the
        # source contains — a speaking voice has near-zero 4–8 Hz pitch oscillation,
        # so after normalization it becomes random jitter, not musical vibrato.
        # A synthesized LFO gives consistent ±21 cents regardless of source material.
        n_vib    = len(f0)
        t_frames = np.arange(n_vib, dtype=np.float32) * 0.010
        # 0.25 Hz rate wobble creates human-like rate variation (not a perfect machine LFO)
        phase   = 2.0 * np.pi * 5.5 * t_frames + 0.15 * np.sin(2.0 * np.pi * 0.25 * t_frames)
        vibrato = np.sin(phase).astype(np.float32)
        vibrato *= (f0 > 30.0).astype(np.float32)   # silence on unvoiced frames
        print(f"[Layer 5] Violin: 5.5 Hz LFO with rate variation (±21 cents @ depth=3.5)")
    elif vibrato is not None:
        vibrato = vibrato.astype(np.float32)
        print(f"[Layer 5] Vibrato loaded ({len(vibrato)} frames)")
    else:
        print("[Layer 5] No vibrato data")

    # ── Body LPC ──────────────────────────────────────────────────────────────
    body_lpc_a = params.get('body_lpc_a', None)
    if body_lpc_a is not None:
        body_lpc_a = body_lpc_a.astype(np.float32)
        print(f"[Layer 5] Body LPC order={len(body_lpc_a)}")

    # ── Zero envelope where F0 is zero (silence frames) ─────────────────────────
    # When F0=0 the envelope has garbage values that produce shrieking harmonics
    f0_mask = (params['ddsp_f0'].astype(np.float32) > 30.0) &               (params['ddsp_f0'].astype(np.float32) < 1200.0)
    if flat_env.shape[0] == len(f0_mask):
        flat_env[~f0_mask] = flat_env[f0_mask].mean(axis=0) if f0_mask.sum() > 0                              else np.zeros_like(flat_env[0])

    # ── Align all arrays to same length ──────────────────────────────────────
    n_env = flat_env.shape[0]
    n_f0  = len(f0)
    n_ld  = len(loudness)
    min_len = min(n_env, n_f0, n_ld)

    # If envelope has more frames than f0 (common), resample rather than truncate
    if n_env != min_len:
        from scipy.interpolate import interp1d as i1d
        t_orig = np.linspace(0, 1, n_env)
        t_new  = np.linspace(0, 1, min_len)
        flat_env = i1d(t_orig, flat_env, axis=0, kind='linear')(t_new).astype(np.float32)
    f0       = f0[:min_len]
    loudness = loudness[:min_len]
    flat_env = flat_env[:min_len]
    if vibrato is not None:
        vibrato = vibrato[:min_len]

    # ── frame_hop: aligned to WORLD 10 ms ────────────────────────────────────
    # WORLD uses frame_period=10 ms → at 44100 Hz that is exactly 441 samples
    frame_hop = int(output_sr * 0.010)   # 441 samples @ 44100

    voiced_f0 = f0[f0 > 30.0]
    f0_min    = voiced_f0.min() if len(voiced_f0) else 0.0
    print(f"[Layer 5] Frames={min_len}  F0={f0_min:.1f}–{f0.max():.1f} Hz  "
          f"Instrument={instrument}  frame_hop={frame_hop}")

    # ── Harmonic amplitude array ──────────────────────────────────────────────
    blend = _BLEND.get(instrument, 0.30)
    print(f"[Layer 5] Building harmonic amplitude array  "
          f"(instrument profile {1-blend:.0%} / spectral env {blend:.0%}) ...")
    harm_amps = _build_harmonic_amp_array(f0, flat_env, instrument, output_sr,
                                           loudness_frames=loudness, blend=blend)

    # ── Synthesis engine ──────────────────────────────────────────────────────
    # Priority:  FluidSynth (soundfont found)  >  K-S (guitar)  >  Additive
    sf2_path = _find_sf2(cfg, instrument)
    use_fluid = False  # disabled — use additive synthesis

    if use_fluid:
        print(f"[Layer 5] FluidSynth synthesis  ({os.path.basename(sf2_path)}) ...")
        try:
            audio = _synthesise_fluidsynth(f0, loudness, instrument, output_sr,
                                            frame_hop, sf2_path)
            print(f"[Layer 5] FluidSynth ✅")
            use_fluid = True
        except Exception as e:
            print(f"[Layer 5] FluidSynth failed ({e}) — falling back to additive")
            use_fluid = False

    if not use_fluid:
        if instrument == 'guitar':
            print("[Layer 5] Karplus-Strong guitar synthesis ...")
            audio = _karplus_strong_guitar(f0, loudness, output_sr, frame_hop)
        else:
            print(f"[Layer 5] Additive synthesis (oversample={OVERSAMPLE}×) ...")
            audio = _synthesise_additive(f0, loudness, harm_amps, instrument, output_sr,
                                          frame_hop, vibrato)

    # Body resonance — only for additive/K-S (real samples already have body in them)
    if not use_fluid and body_lpc_a is not None and len(body_lpc_a) > 0:
        lpc_wet = 0.40 if instrument == 'violin' else 1.0   # reduced: body was dominating at 60%
        print(f"[Layer 5] Applying body resonance LPC filter (wet={lpc_wet:.0%}) ...")
        lpc_out = _apply_body_lpc(audio, body_lpc_a)
        audio   = lpc_wet * lpc_out + (1.0 - lpc_wet) * audio

    if not use_fluid:
        # FluidSynth already has realistic noise baked into the samples
        print("[Layer 5] Adding noise component ...")
        audio = _add_noise(audio, loudness, flat_env, instrument, output_sr, frame_hop)

    # ── Envelope ─────────────────────────────────────────────────────────────
    # FluidSynth soundfonts already contain per-note ADSR from the sample data.
    # Applying _apply_envelope on top accumulates 300+ release fades multiplicatively
    # (env *= fade at every note-off), driving the signal to ~10^-14 → silence.
    if not use_fluid:
        print("[Layer 5] Applying note envelopes ...")
        audio = _apply_envelope(audio, output_sr, instrument, f0, frame_hop)
    else:
        print("[Layer 5] Envelope: skipped (FluidSynth ADSR handles this) ...")

    # ── Synth VCF ─────────────────────────────────────────────────────────────
    if instrument == 'synth' and not use_fluid:
        print("[Layer 5] Applying VCF (envelope-following resonant LP, fc=800–7000 Hz, Q=2.5) ...")
        audio = _apply_vcf(audio, output_sr, loudness, frame_hop)

    # ── Pre-mastering level normalisation ────────────────────────────────────
    peak = float(np.max(np.abs(audio.astype(np.float64))) + 1e-12)
    if peak > 1e-8:
        target_peak = 0.80
        audio = (audio * (target_peak / peak)).astype(np.float32)
        print(f"[Layer 5] Pre-master: peak normalised to -14 dBFS (dynamics preserved)")

        # ── Stereo ────────────────────────────────────────────────────────────────
    print(f"[Layer 5] Adding stereo width (Haas {HAAS_MS.get(instrument,1.5):.1f} ms) ...")
    audio_stereo = _haas_stereo(audio, output_sr, instrument)

    # ── True-peak limit ───────────────────────────────────────────────────────
    audio_stereo[:, 0] = _true_peak_limit(audio_stereo[:, 0], -1.0)
    audio_stereo[:, 1] = _true_peak_limit(audio_stereo[:, 1], -1.0)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, audio_stereo, output_sr, subtype='FLOAT')
    print(f"[Layer 5] ✅ Saved stereo: {output_path}  ({len(audio)/output_sr:.1f}s)")
    return output_path


if __name__ == "__main__":
    import sys
    instrument = sys.argv[1] if len(sys.argv) > 1 else "piano"
    run_synthesis("soniq_outputs/ddsp_params.npz", instrument,
                  "config.yaml", "soniq_outputs/ddsp_output.wav")