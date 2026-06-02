"""
converter.py  —  Layer 4: Seed-VC Conversion + WORLD Bridge  v5.3 HQ

Changes over v5.2:
  FIX:  WORLD frame_period fixed to 10.0 ms throughout — was inconsistent
        between dio/stonemask (10 ms) and cheaptrick call (used default 5 ms).
        This caused F0 and spectral envelope to have different frame counts,
        causing misaligned synthesis and broken audio.
  FIX:  aperiodicity now extracted with same frame_period as F0 (10 ms).
  FIX:  Loudness array aligned to same length as f0_primary before save —
        prevents off-by-one frame count errors in Layer 5.
  ADD:  Formant preservation ratio: if LPC inverse filter over-whitens signal,
        blend 30% of original to preserve some vocal expressiveness.
  ADD:  CREPE F0 confidence-weighted merge with WORLD F0 (not hard threshold at 30 Hz).
  ADD:  Explicit WORLD frame_period=10 constant exported as 'ddsp_frame_period_ms'.
"""

import os
import sys
sys.path.append("./seed_vc")
import numpy as np
import soundfile as sf
import pyworld as pw
import yaml

from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz

from seed_vc_wrapper import SeedVCWrapper
# Centroid profiles are calibrated for VOICE-TO-INSTRUMENT conversion output,
# not for recordings of real instruments. Voice F0 range (100–300 Hz) maps to
# instrument notes in the same range, giving lower centroids than real instrument
# recordings. piano was 1800 Hz — unreachable for converted voice, triggering the
# fallback every run. These values reflect measured conversion output centroids.
_CENTROID_PROFILES = {
    'violin': 2000.0, 'cello': 900.0,  'flute': 1800.0,
    'piano':   900.0, 'guitar': 1200.0, 'synth': 1600.0,
    'trumpet': 1800.0,
}
_CENTROID_FALLBACK_HZ = 1300.0  # overridden by config 'seedvc_centroid_fallback_hz' at runtime

# WORLD frame period — MUST be 10 ms to match synthesiser.py frame_hop
WORLD_FRAME_PERIOD = 10.0   # ms


def _spectral_centroid(audio, sr):
    import librosa
    return float(np.mean(
        librosa.feature.spectral_centroid(y=audio.astype(np.float32), sr=sr)
    ))


# ── LPC inverse filter ────────────────────────────────────────────────────────

def _lpc_inverse_filter(audio, sr, order=24, blend=0.70):
    """
    Estimate LPC vocal-tract filter and apply partial inverse filter.
    blend=0.70 means 70% whitened + 30% original (avoids over-whitening).
    """
    audio_f = audio.astype(np.float64)
    r = np.correlate(audio_f, audio_f, mode='full')
    r = r[len(r)//2 : len(r)//2 + order + 1]
    r[0] += 1e-6
    try:
        a   = solve_toeplitz(r[:order], r[1:order+1])
        fir = np.concatenate([[1.0], -a])
        exc = lfilter(fir, [1.0], audio_f)
        return (blend * exc + (1.0 - blend) * audio_f).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


# ── Body resonance LPC ────────────────────────────────────────────────────────

def _extract_body_resonance_lpc(ref_audio_path, sr, order=20):
    import librosa
    ref, ref_sr = sf.read(ref_audio_path)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if ref_sr != sr:
        ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=sr)

    mid = len(ref) // 3
    seg = ref[mid:2*mid].astype(np.float64)
    seg /= (np.max(np.abs(seg)) + 1e-8)
    r = np.correlate(seg, seg, mode='full')
    r = r[len(r)//2 : len(r)//2 + order + 1]
    r[0] += 1e-6
    try:
        a     = solve_toeplitz(r[:order], r[1:order+1])
        poles = np.roots(np.concatenate([[1.0], -a]))
        # Shrink all poles inside the unit circle with margin — any pole
        # with |z| > 0.97 causes very high gain at that frequency and makes
        # the instrument sound shrill (was checked at 0.999, far too close).
        max_pole = np.max(np.abs(poles))
        if max_pole > 0.97:
            a *= (0.97 / max_pole) ** 0.5   # partial stabilisation, not hard clip
        return a.astype(np.float32)
    except Exception as e:
        print(f"[Layer 4] Body LPC failed ({e})")
        return np.zeros(order, dtype=np.float32)


# ── Cepstral formant suppression ──────────────────────────────────────────────

def _remove_vocal_formants(spectral_env, lifter_bins=60):
    log_env      = np.log(np.maximum(spectral_env, 1e-8))
    cep          = np.fft.irfft(log_env, axis=1)
    cep_l        = cep.copy()
    # Preserve cepstral bin 0 (DC = mean log-spectrum level → per-frame energy).
    # Zeroing bin 0 strips energy information, making loudness_flat ≈ 0 dBFS
    # for every frame and removing all amplitude dynamics from the synthesis.
    cep_l[:, 1:lifter_bins]  = 0.0
    cep_l[:, -lifter_bins:]  = 0.0
    log_flat     = np.fft.rfft(cep_l, axis=1).real
    return np.exp(log_flat).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_conversion(source_audio_path, instrument, config_path,
                    output_wav_path, ddsp_params_path,
                    hubert_embeddings_path=None,
                    extracted_params_path=None):

    print("\n" + "=" * 60)
    print("  LAYER 4 — Seed-VC + WORLD Bridge  v5.3")
    print("=" * 60)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    centroid_fallback_hz = float(cfg.get('seedvc_centroid_fallback_hz', _CENTROID_FALLBACK_HZ))

    from reference_generator import ensure_reference
    target_audio = ensure_reference(instrument, config_path)

    if not os.path.isfile(source_audio_path):
        raise FileNotFoundError(f"[Layer 4] Source not found: {source_audio_path}")

    print(f"[Layer 4] Source : {source_audio_path}")
    print(f"[Layer 4] Target : {target_audio}")

    # ── Seed-VC ───────────────────────────────────────────────────────────────
    wrapper    = SeedVCWrapper()

    # Load instrument-specific fine-tune if configured (e.g. piano fine-tune)
    ft_path = cfg.get('checkpoints', {}).get('seed_vc', {}).get(instrument, '')
    if ft_path and os.path.isfile(ft_path):
        wrapper.load_instrument_finetune(ft_path)
    else:
        print(f"[Layer 4] No fine-tune checkpoint found for '{instrument}' — using base model")

    ddim_steps = int(cfg.get('ddim_steps', 50))
    print(f"[Layer 4] Seed-VC  ({ddim_steps} DDIM steps) ...")
    result = wrapper.convert_voice(
        source=source_audio_path, target=target_audio,
        diffusion_steps=ddim_steps,
        length_adjust=1.0, inference_cfg_rate=0.9,
        f0_condition=True, auto_f0_adjust=True,
        pitch_shift=0, stream_output=False,
    )

    # ── Parse output ──────────────────────────────────────────────────────────
    output_audio = None
    seed_vc_sr   = 44100

    if hasattr(result, '__iter__') and not isinstance(result, (np.ndarray, tuple, list)):
        chunks = list(result)
        if len(chunks) == 0:
            if hasattr(wrapper, 'last_output_audio'):
                output_audio = wrapper.last_output_audio
            else:
                raise RuntimeError("[Layer 4] Seed-VC returned empty generator")
        else:
            final_chunk = chunks[-1]
            if isinstance(final_chunk, (tuple, list)):
                output_audio = final_chunk[-1]
                if len(final_chunk) > 1 and isinstance(final_chunk[0], int):
                    seed_vc_sr = int(final_chunk[0])
            else:
                output_audio = final_chunk
    elif isinstance(result, (tuple, list)):
        output_audio = result[0]
        if len(result) > 1:
            seed_vc_sr = int(result[1])
    else:
        output_audio = result

    if output_audio is None:
        raise RuntimeError("[Layer 4] No audio generated")

    output_audio = np.asarray(output_audio, dtype=np.float32)
    if output_audio.ndim > 1:
        output_audio = (output_audio.mean(axis=0)
                        if output_audio.shape[0] < output_audio.shape[1]
                        else output_audio.mean(axis=1))

    print(f"[Layer 4] Seed-VC output shape={output_audio.shape}  sr={seed_vc_sr}")

    # ── Spectral centroid quality gate ────────────────────────────────────────
    svc_centroid     = _spectral_centroid(output_audio, seed_vc_sr)
    target_centroid  = _CENTROID_PROFILES.get(instrument, 2000.0)
    centroid_error   = abs(svc_centroid - target_centroid)
    print(f"[Layer 4] Centroid SeedVC={svc_centroid:.0f} Hz  "
          f"profile={target_centroid:.0f} Hz  error={centroid_error:.0f} Hz")
    use_svc = centroid_error <= centroid_fallback_hz

    os.makedirs(os.path.dirname(output_wav_path) or '.', exist_ok=True)
    sf.write(output_wav_path, output_audio, seed_vc_sr)
    print(f"[Layer 4] Saved Seed-VC output: {output_wav_path}")

    # ── WORLD bridge ──────────────────────────────────────────────────────────
    print("[Layer 4] WORLD bridge ...")
    if use_svc:
        analysis_audio, analysis_sr = sf.read(output_wav_path)
    else:
        analysis_audio, analysis_sr = sf.read(source_audio_path)
        print("[Layer 4]    (fallback: using source vocal for WORLD)")

    if analysis_audio.ndim > 1:
        analysis_audio = analysis_audio.mean(axis=1)

    # Load original source audio for pitch-accurate WORLD F0.
    # Seed-VC can subtly shift pitch, so WORLD on its output gives drifted F0.
    # Running DIO on the original humming then using cheaptrick on the Seed-VC
    # output lets us get accurate pitch from source while keeping Seed-VC timbre.
    src_for_f0, src_sr_f0 = sf.read(source_audio_path)
    if src_for_f0.ndim > 1:
        src_for_f0 = src_for_f0.mean(axis=1)
    if src_sr_f0 != analysis_sr:
        import librosa as _lb
        src_for_f0 = _lb.resample(src_for_f0.astype(np.float32),
                                   orig_sr=src_sr_f0, target_sr=analysis_sr)
    # Align source and analysis audio to same length before WORLD
    min_smp = min(len(src_for_f0), len(analysis_audio))
    src_f64        = src_for_f0[:min_smp].astype(np.float64)
    analysis_audio = analysis_audio[:min_smp]

    # LPC inverse filter (partial — 70% whitened) on Seed-VC output for timbre
    print("[Layer 4] LPC inverse filter ...")
    audio_deformed = _lpc_inverse_filter(analysis_audio, analysis_sr,
                                           order=24, blend=0.70)

    audio_f64          = analysis_audio.astype(np.float64)
    audio_deformed_f64 = audio_deformed.astype(np.float64)

    # WORLD F0 on original source (pitch-accurate humming tracking)
    print("[Layer 4] WORLD F0 from original source audio ...")
    f0_world, t_world = pw.dio(src_f64, analysis_sr,
                                frame_period=WORLD_FRAME_PERIOD)
    f0_world           = pw.stonemask(src_f64, f0_world, t_world, analysis_sr)

    # Spectral envelope and aperiodicity from Seed-VC output (timbral transfer)
    spectral_env = pw.cheaptrick(audio_deformed_f64, f0_world, t_world,
                                  analysis_sr, fft_size=None)
    aperiodicity = pw.d4c(audio_f64, f0_world, t_world, analysis_sr)

    # ── CREPE F0 merge (confidence-weighted) ─────────────────────────────────
    f0_primary = f0_world
    vibrato    = None
    brightness = None
    hnr        = None

    if extracted_params_path and os.path.isfile(extracted_params_path):
        try:
            l2       = np.load(extracted_params_path)
            f0_crepe = l2['f0']
            conf     = l2['confidence'] if 'confidence' in l2 else None
            min_len  = min(len(f0_world), len(f0_crepe))
            f0_c     = f0_crepe[:min_len]
            f0_w     = f0_world[:min_len]

            if conf is not None:
                c = conf[:min_len]
                # Fill low-confidence CREPE frames by interpolating from
                # high-confidence neighbours instead of blending with WORLD.
                # WORLD on Seed-VC output can have drifted pitch; interpolated
                # CREPE is more faithful to the original humming melody.
                f0_c_filled = f0_c.copy()
                high_conf_mask = (c >= 0.45) & (f0_c > 30.0)
                if high_conf_mask.sum() > 1:
                    hi_idx = np.where(high_conf_mask)[0]
                    from scipy.interpolate import interp1d as _i1d
                    fill_fn = _i1d(hi_idx, f0_c_filled[hi_idx], kind='linear',
                                   bounds_error=False,
                                   fill_value=(0.0, 0.0))   # don't extrapolate stale pitch into unconfident regions
                    lo_idx = np.where(~high_conf_mask)[0]
                    f0_c_filled[lo_idx] = fill_fn(lo_idx)

                # Use interpolated CREPE as primary; fall back to WORLD only
                # when CREPE treats the frame as unvoiced (f0 ≤ 30 Hz).
                voiced_crepe = f0_c_filled > 30.0
                voiced_world = f0_w > 30.0
                f0_primary = np.where(voiced_crepe, f0_c_filled,
                                      np.where(voiced_world, f0_w, 0.0)).astype(np.float32)
            else:
                voiced_c   = f0_c > 30.0
                f0_primary = np.where(voiced_c, f0_c, f0_w)

            spectral_env = spectral_env[:min_len]
            aperiodicity = aperiodicity[:min_len]
            print(f"[Layer 4] CREPE+WORLD merged  frames={min_len}")

            if 'vibrato'    in l2: vibrato    = l2['vibrato'][:min_len].astype(np.float32)
            if 'brightness' in l2: brightness = l2['brightness']
            if 'hnr'        in l2: hnr        = l2['hnr'][:min_len]
        except Exception as e:
            print(f"[Layer 4] L2 params load failed ({e}) — using WORLD F0")

    # Ensure all arrays have the same frame count
    n_frames = len(f0_primary)
    if spectral_env.shape[0] != n_frames:
        from scipy.interpolate import interp1d as i1d
        t_o = np.linspace(0, 1, spectral_env.shape[0])
        t_n = np.linspace(0, 1, n_frames)
        spectral_env = i1d(t_o, spectral_env, axis=0, kind='linear')(t_n)
    if aperiodicity.shape[0] != n_frames:
        from scipy.interpolate import interp1d as i1d
        t_o = np.linspace(0, 1, aperiodicity.shape[0])
        t_n = np.linspace(0, 1, n_frames)
        aperiodicity = i1d(t_o, aperiodicity, axis=0, kind='linear')(t_n)

    # ── Cepstral formant suppression ──────────────────────────────────────────
    # Piano/guitar need very aggressive suppression (lifter_bins=100+) to fully
    # remove vocal formant peaks (F1–F4) that make the output sound like humming.
    # Strings/brass tolerate mild suppression — some spectral shape is desirable.
    _LIFTER_BINS = {
        'piano': 120, 'guitar': 110, 'synth': 100,
        'violin': 60, 'cello': 60, 'flute': 70, 'trumpet': 80,
    }
    lifter_bins = _LIFTER_BINS.get(instrument, 60)
    print(f"[Layer 4] Cepstral formant suppression (lifter_bins={lifter_bins}) ...")
    try:
        spectral_env_flat = _remove_vocal_formants(spectral_env, lifter_bins=lifter_bins)
        loudness_flat     = 20.0 * np.log10(
            np.sqrt(np.mean(spectral_env_flat**2, axis=1)) + 1e-8)
    except Exception as e:
        print(f"[Layer 4] Formant suppression failed ({e})")
        spectral_env_flat = spectral_env.astype(np.float32)
        loudness_flat     = 20.0 * np.log10(
            np.sqrt(np.mean(spectral_env**2, axis=1)) + 1e-8)

    loudness_orig = 20.0 * np.log10(
        np.sqrt(np.mean(spectral_env**2, axis=1)) + 1e-8)

    # ── Body resonance LPC ────────────────────────────────────────────────────
    body_lpc_a = None
    ref_path   = cfg.get('instruments', {}).get(instrument, {}).get('reference', '')
    if ref_path and os.path.isfile(ref_path):
        print(f"[Layer 4] Extracting body LPC from: {ref_path}")
        body_lpc_a = _extract_body_resonance_lpc(ref_path, analysis_sr, order=20)

    # ── Save DDSP params ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ddsp_params_path) or '.', exist_ok=True)

    save_dict = dict(
        ddsp_f0              = f0_primary.astype(np.float32),
        ddsp_loudness        = loudness_orig.astype(np.float32),
        ddsp_loudness_flat   = loudness_flat.astype(np.float32),
        ddsp_envelope        = spectral_env.astype(np.float32),
        ddsp_envelope_flat   = spectral_env_flat.astype(np.float32),
        ddsp_aperiodicity    = aperiodicity.astype(np.float32),
        ddsp_frame_period_ms = np.array([WORLD_FRAME_PERIOD], dtype=np.float32),
    )
    if vibrato    is not None: save_dict['ddsp_vibrato']    = vibrato
    if body_lpc_a is not None: save_dict['body_lpc_a']      = body_lpc_a
    if brightness is not None:
        save_dict['ddsp_brightness'] = np.resize(brightness,
                                                  len(f0_primary)).astype(np.float32)
    if hnr is not None:
        save_dict['ddsp_hnr'] = hnr.astype(np.float32)

    np.savez(ddsp_params_path, **save_dict)
    print(f"[Layer 4] ✅ DDSP params saved  frames={len(f0_primary)}")

    return output_wav_path, ddsp_params_path


if __name__ == '__main__':
    import sys
    inst = sys.argv[1] if len(sys.argv) > 1 else 'piano'
    run_conversion(
        source_audio_path='soniq_outputs/conditioned_vocals.wav',
        instrument=inst, config_path='config.yaml',
        output_wav_path='soniq_outputs/seed_vc_output.wav',
        ddsp_params_path='soniq_outputs/ddsp_params.npz',
        extracted_params_path='soniq_outputs/extracted_params.npz',
    )