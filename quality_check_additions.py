"""
quality_check_additions.py
──────────────────────────
Paste these functions into quality_check.py (after check_layer3).
They complete the QC coverage for Layers 4, 5, and 6.
"""

import numpy as np
import soundfile as sf


# ── Layer 4: Seed-VC output quality ──────────────────────────────────────────
def check_layer4(
    seed_vc_wav_path: str,
    extracted_params_path: str,
) -> str:
    """
    Layer 4: Check Seed-VC output quality.
    Metrics:
        snr           — signal-to-noise ratio of the output waveform
        f0_correlation — Pearson correlation between input F0 (Layer 2) and
                         re-analysed F0 from Seed-VC output (WORLD bridge)
    """
    from quality_check import check_layer   # avoid circular import in standalone use

    # Load Seed-VC output
    audio, sr = sf.read(seed_vc_wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # SNR: ratio of signal power to estimated noise floor
    # Simple estimate: top 10% of RMS windows vs bottom 10%
    hop = int(sr * 0.02)  # 20 ms windows
    n_frames = len(audio) // hop
    rms_per_frame = np.array([
        np.sqrt(np.mean(audio[i*hop:(i+1)*hop]**2))
        for i in range(n_frames)
    ])
    rms_per_frame = rms_per_frame[rms_per_frame > 0]
    if len(rms_per_frame) < 4:
        snr = 0.0
    else:
        signal_power = np.percentile(rms_per_frame, 90)
        noise_floor  = np.percentile(rms_per_frame, 10) + 1e-10
        snr = float(20.0 * np.log10(signal_power / noise_floor))

    # F0 correlation: compare Layer 2 CREPE F0 with WORLD F0 from bridge
    f0_correlation = None
    try:
        import pyworld as pw
        audio_f64 = audio.astype(np.float64)
        f0_world, t = pw.dio(audio_f64, sr, frame_period=10.0)
        f0_world    = pw.stonemask(audio_f64, f0_world, t, sr)

        params   = np.load(extracted_params_path)
        f0_crepe = params['f0']

        min_len = min(len(f0_world), len(f0_crepe))
        voiced  = (f0_world[:min_len] > 50.0) & (f0_crepe[:min_len] > 50.0)
        if voiced.sum() > 10:
            f0_corr = np.corrcoef(
                f0_world[:min_len][voiced],
                f0_crepe[:min_len][voiced],
            )[0, 1]
            f0_correlation = float(f0_corr)
    except Exception as e:
        print(f"[QC]  Layer 4: F0 correlation skipped ({e})")

    metrics = {'snr': snr}
    if f0_correlation is not None:
        metrics['f0_correlation'] = f0_correlation

    print(f"[QC]  Layer 4: SNR={snr:.1f} dB  "
          f"F0_corr={f0_correlation:.3f}" if f0_correlation else
          f"[QC]  Layer 4: SNR={snr:.1f} dB  F0_corr=n/a")
    return check_layer(4, metrics)


# ── Layer 5: DDSP synthesis quality ──────────────────────────────────────────
def check_layer5(
    ddsp_output_path: str,
    reference_audio_path: str,
) -> str:
    """
    Layer 5: Check DDSP output spectral quality vs the instrument reference.
    Metric:
        spectral_centroid_error_hz — absolute difference between DDSP output
                                     and reference instrument recording spectral centroid
    """
    from quality_check import check_layer
    import librosa

    ddsp_audio, ddsp_sr = sf.read(ddsp_output_path)
    if ddsp_audio.ndim > 1:
        ddsp_audio = ddsp_audio.mean(axis=1)

    ref_audio, ref_sr = sf.read(reference_audio_path)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    # Resample reference to same SR as DDSP output if needed
    if ref_sr != ddsp_sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=ddsp_sr)

    ddsp_centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=ddsp_audio.astype(np.float32), sr=ddsp_sr
    )))
    ref_centroid  = float(np.mean(librosa.feature.spectral_centroid(
        y=ref_audio.astype(np.float32), sr=ddsp_sr
    )))
    error_hz = abs(ddsp_centroid - ref_centroid)

    metrics = {'spectral_centroid_error_hz': error_hz}
    print(f"[QC]  Layer 5: spectral_centroid  "
          f"ddsp={ddsp_centroid:.0f} Hz  ref={ref_centroid:.0f} Hz  "
          f"error={error_hz:.0f} Hz")
    return check_layer(5, metrics)


# ── Layer 6: Mastering / loudness quality ────────────────────────────────────
def check_layer6(
    final_output_path: str,
    config_path: str,
) -> str:
    """
    Layer 6: Verify final output meets loudness target.
    Metric:
        lufs_tolerance — absolute deviation from target_lufs (must be < 0.5)
    """
    from quality_check import check_layer
    import pyloudnorm as pyln
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    target_lufs = float(cfg.get('target_lufs', -14.0))

    audio, sr = sf.read(final_output_path)
    meter = pyln.Meter(sr)
    measured_lufs = meter.integrated_loudness(
        audio if audio.ndim > 1 else audio[:, np.newaxis]
    )
    deviation = abs(measured_lufs - target_lufs)

    metrics = {'lufs_tolerance': deviation}
    print(f"[QC]  Layer 6: LUFS  measured={measured_lufs:.2f}  "
          f"target={target_lufs:.2f}  deviation={deviation:.3f}")
    return check_layer(6, metrics)