import numpy as np
import soundfile as sf
import pyworld as pw
import librosa
import crepe

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Utilities
# ============================================================

def smooth_f0(f0):

    f0 = gaussian_filter1d(
        f0,
        sigma=1.2
    )

    return f0.astype(np.float32)


def stabilize_pitch(frequency):

    # --------------------------------------------------------
    # Initial smoothing
    # --------------------------------------------------------

    frequency = smooth_f0(frequency)

    # --------------------------------------------------------
    # Intelligent spike suppression
    # --------------------------------------------------------

    for i in range(1, len(frequency)):

        prev = frequency[i - 1]
        curr = frequency[i]

        if prev <= 0 or curr <= 0:
            continue

        jump_cents = abs(
            1200 * np.log2(curr / prev)
        )

        # Suppress only true tracker glitches (octave errors, wild spikes).
        # 150 cents suppressed jumps > minor 2nd, killing every melodic interval
        # larger than ~1.5 semitones — e.g. a major 2nd (200 c) or minor 3rd
        # (300 c) was silently replaced with the previous frame pitch, causing
        # the synthesized melody to "freeze" at each note instead of jumping.
        # CREPE with viterbi=True already avoids most octave errors; 800 cents
        # catches only genuine ≥ 8-semitone glitch spikes.
        if jump_cents > 800:

            frequency[i] = prev

    # --------------------------------------------------------
    # Final expressive smoothing
    # --------------------------------------------------------

    frequency = gaussian_filter1d(
        frequency,
        sigma=1.0
    )

    return frequency.astype(np.float32)


def compute_f0_continuity(f0):

    diff = np.diff(f0)

    continuity = np.mean(np.abs(diff))

    return continuity


def extract_vibrato(f0):
    # Vibrato is a 5–7 Hz quasi-periodic oscillation of F0, not its derivative.
    # np.gradient gave the frame-to-frame slope, which is noise-dominated and
    # unrelated to the actual vibrato modulation fed into synthesis.
    voiced = f0 > 30.0
    if voiced.sum() < 10:
        return np.zeros_like(f0, dtype=np.float32)

    # Work in log-F0 (cents-proportional); fill unvoiced with interpolation
    log_f0 = np.where(voiced, np.log(np.maximum(f0, 1.0)), np.nan)
    idx = np.arange(len(log_f0))
    valid = np.isfinite(log_f0)
    if valid.sum() > 1:
        log_f0 = np.interp(idx, idx[valid], log_f0[valid])

    # CREPE step_size=10 ms → frame rate = 100 Hz. Bandpass 4–8 Hz for vibrato.
    sr_frames = 100.0
    lo = 4.0 / (sr_frames / 2.0)
    hi = min(8.0 / (sr_frames / 2.0), 0.999)
    sos = butter(2, [lo, hi], btype='band', output='sos')
    vibrato = sosfiltfilt(sos, log_f0)
    # Zero out unvoiced regions
    vibrato = np.where(voiced, vibrato, 0.0)
    return vibrato.astype(np.float32)


def extract_energy(audio):

    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    return rms.astype(np.float32)


def detect_onsets(audio, sr):

    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        backtrack=True
    )

    return onset_frames.astype(np.int32)


def articulation_features(audio, sr):

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    spectral_flux = librosa.onset.onset_strength(
        y=audio,
        sr=sr
    )

    return (
        zcr.astype(np.float32),
        spectral_flux.astype(np.float32)
    )


# ============================================================
# Main Layer 2
# ============================================================

def run_extraction(
    input_path,
    output_path
):

    print("\n============================================================")
    print("  LAYER 2 — Docs-Grade Acoustic Intelligence")
    print("============================================================")

    # --------------------------------------------------------
    # Load Audio
    # --------------------------------------------------------

    audio, sr = sf.read(input_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print(f"[Layer 2] Input: {input_path}")
    print(f"[Layer 2] SR: {sr}")
    print(f"[Layer 2] Duration: {len(audio)/sr:.2f}s")

    # --------------------------------------------------------
    # WORLD Analysis
    # --------------------------------------------------------

    print("[Layer 2] Running WORLD decomposition...")

    _f0, t = pw.harvest(
        audio.astype(np.float64),
        sr
    )

    sp = pw.cheaptrick(
        audio.astype(np.float64),
        _f0,
        t,
        sr
    )

    ap = pw.d4c(
        audio.astype(np.float64),
        _f0,
        t,
        sr
    )

    print(f"[Layer 2] WORLD done | sp={sp.shape}")

    # --------------------------------------------------------
    # CREPE Pitch Tracking
    # --------------------------------------------------------

    print("[Layer 2] Running CREPE...")

    time, frequency, confidence, activation = crepe.predict(
        audio,
        sr,
        viterbi=True,
        step_size=10,
        verbose=0
    )

    frequency = np.nan_to_num(frequency)
    # Clip at 1200 Hz (soprano ceiling) — 700 Hz silenced any note above E5,
    # which is well within normal humming range. CREPE viterbi=True already
    # rejects octave jumps, so this guard is only for extreme tracker failures.
    frequency = np.where(frequency > 1200.0, 0.0, frequency)

    print(f"[Layer 2] CREPE done | frames={len(frequency)}")

    # --------------------------------------------------------
    # Intelligent Pitch Stabilization
    # --------------------------------------------------------

    frequency = stabilize_pitch(frequency)


    # --------------------------------------------------------
    # Vibrato Extraction
    # --------------------------------------------------------

    vibrato = extract_vibrato(frequency)

    # --------------------------------------------------------
    # Harmonic Continuity
    # --------------------------------------------------------

    continuity = compute_f0_continuity(frequency)

    # --------------------------------------------------------
    # Energy Envelope
    # --------------------------------------------------------

    energy = extract_energy(audio)

    # --------------------------------------------------------
    # Onset Detection
    # --------------------------------------------------------

    onsets = detect_onsets(audio, sr)

    # --------------------------------------------------------
    # Articulation Features
    # --------------------------------------------------------

    zcr, spectral_flux = articulation_features(
        audio,
        sr
    )

    # --------------------------------------------------------
    # DDSP Conditioning
    # --------------------------------------------------------

    harmonic_energy = np.mean(
        sp,
        axis=1
    )

    noise_energy = np.mean(
        ap,
        axis=1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(

        output_path,

        # compatibility
        f0=frequency,

        # WORLD
        f0_world=_f0,
        spectral_envelope=sp,
        aperiodicity=ap,

        # CREPE
        f0_crepe=frequency,
        confidence=confidence,

        # expressive
        vibrato=vibrato,
        continuity=continuity,
        energy=energy,

        # articulation
        onsets=onsets,
        zcr=zcr,
        spectral_flux=spectral_flux,

        # DDSP-ready
        harmonic_energy=harmonic_energy,
        noise_energy=noise_energy
    )

    print(f"[Layer 2] Saved: {output_path}")

    print(f"[Layer 2] Frames: {len(frequency)}")
    print(f"[Layer 2] Onsets detected: {len(onsets)}")

    print(
        f"[Layer 2] Harmonic continuity: "
        f"{continuity:.3f}"
    )

    return output_path


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    run_extraction(
        "soniq_outputs/conditioned_vocals.wav",
        "soniq_outputs/extracted_params.npz"
    )
