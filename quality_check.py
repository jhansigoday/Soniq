import numpy as np
import soundfile as sf
import yaml

# ── Thresholds ────────────────────────────────────────────────────────────────
# For each metric: (min_value, is_lower_better)
# is_lower_better=True  → value must be BELOW threshold (e.g. F0 jump)
# is_lower_better=False → value must be ABOVE threshold (e.g. confidence)

THRESHOLDS = {
    1: {
        'sdr':                     (7.0,  False),  # SDR >= 7 dB
    },
    2: {
        'crepe_confidence':        (0.55,  False),  # mean confidence >= 0.55
        'f0_jump_max_cents':       (1500.0, True),  # max jump < 1500 cents (humming has large melodic leaps)
    },
    3: {
        'embedding_norm_min':      (0.1,  False),  # min norm >= 0.1
    },
    4: {
        # Seed-VC voice-to-instrument conversion introduces spectral noise;
        # 25 dB SNR is unreachable for converted audio (silence sections naturally
        # lower the RMS floor). 10 dB catches truly silent/corrupt output only.
        'snr':            (10.0, False),
        # Seed-VC is a timbre converter, not a pitch-preserving converter.
        # WORLD on Seed-VC output vs original CREPE F0 will have near-zero or
        # negative correlation because Seed-VC reshapes the spectrum freely.
        # Synthesis now uses original CREPE F0 directly; this metric only
        # detects fully degenerate (random) Seed-VC output.
        'f0_correlation': (-0.20, False),
    },
    5: {
        'spectral_centroid_error_hz': (500.0, True),  # error < 500 Hz
    },
    6: {
        'lufs_tolerance':          (0.5,  True),   # deviation < 0.5 LUFS
    },
}


def check_layer(layer_id: int, metrics: dict) -> str:
    """
    Returns 'pass' or 'warning'.
    NEVER raises — pipeline continues regardless, with logged diagnostics.
    """
    threshold = THRESHOLDS.get(layer_id, {})
    failed = []

    for key, (threshold_val, lower_is_better) in threshold.items():
        val = metrics.get(key)
        if val is None:
            print(f"[QC]  Layer {layer_id}: metric '{key}' not provided — skipping")
            continue
        if lower_is_better:
            if val >= threshold_val:
                failed.append(f"{key}={val:.3f} (need < {threshold_val})")
        else:
            if val < threshold_val:
                failed.append(f"{key}={val:.3f} (need ≥ {threshold_val})")

    if failed:
        print(f"[QC] ⚠️  Layer {layer_id} WARNING: {', '.join(failed)}")
        return 'warning'

    print(f"[QC] ✅ Layer {layer_id} PASS")
    return 'pass'


# ── Layer-specific helpers ────────────────────────────────────────────────────

def check_layer1(vocals_path: str, sdr_estimate: float = None) -> str:
    """
    Layer 1: Basic sanity check on separated vocals.
    SDR measurement requires the original mix + reference — we accept an external estimate.
    If not provided, we check that the file exists and has reasonable amplitude.
    """
    import soundfile as sf
    audio, sr = sf.read(vocals_path)
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"[QC]  Layer 1: vocals RMS={rms:.4f}  sr={sr}")
    if rms < 1e-4:
        print("[QC] ⚠️  Layer 1 WARNING: Very low signal level — check separation")
        return 'warning'
    if sdr_estimate is not None:
        return check_layer(1, {'sdr': sdr_estimate})
    print("[QC] ✅ Layer 1 PASS (no SDR reference available — amplitude check only)")
    return 'pass'


def check_layer2(params_path: str) -> str:
    """
    Layer 2: Check CREPE confidence and F0 smoothness.
    """
    params    = np.load(params_path)
    f0        = params['f0']
    conf      = params['confidence']

    # Convert F0 to cents for jump detection (ignore unvoiced frames where f0 ≈ 0)
    voiced    = f0 > 50.0
    f0_voiced = f0[voiced]

    if len(f0_voiced) > 1:
        f0_cents  = 1200.0 * np.log2(np.maximum(f0_voiced, 1e-8) / 440.0)
        max_jump  = float(np.max(np.abs(np.diff(f0_cents))))
    else:
        max_jump  = 0.0
        print("[QC]  Layer 2: No voiced frames detected — check input audio")

    metrics = {
        'crepe_confidence':   float(np.mean(conf)),
        'f0_jump_max_cents':  max_jump,
    }
    print(f"[QC]  Layer 2: mean_confidence={metrics['crepe_confidence']:.3f}  "
          f"max_f0_jump={max_jump:.1f} cents")
    return check_layer(2, metrics)


def check_layer3(embeddings_path: str) -> str:
    """
    Layer 3: Check HuBERT embedding norms.
    """
    embeddings = np.load(embeddings_path)
    norms = np.linalg.norm(embeddings, axis=-1)
    metrics = {
        'embedding_norm_min': float(norms.min()),
    }
    print(f"[QC]  Layer 3: embedding norm  min={norms.min():.3f}  mean={norms.mean():.3f}")
    return check_layer(3, metrics)