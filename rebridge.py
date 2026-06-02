"""
rebridge.py  —  Regenerate ddsp_params.npz without re-running Seed-VC (~30 s)

Root-cause fix applied here:
  The CREPE fill_value was (last_good_f0, last_good_f0), so any frame in the
  second half where CREPE confidence < 0.45 was extrapolated to the last
  confident pitch from the first half — a constant stale value.  Since
  voiced_crepe was True for those frames, they overrode WORLD, causing the
  second half to play one repeated wrong note regardless of the actual humming.

  Fix: fill_value=(0.0, 0.0) — unconfident out-of-range frames get f0=0,
  voiced_crepe becomes False, and the merge falls back to WORLD's f0 from
  the original humming (or stays silent if WORLD also has no pitch there).

Usage:
    python rebridge.py [instrument]
    python rebridge.py piano
"""

import os
import sys
import numpy as np
import soundfile as sf
import pyworld as pw
import yaml
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz
from scipy.interpolate import interp1d as i1d

WORLD_FRAME_PERIOD = 10.0  # ms — must match synthesiser.py frame_hop


def _lpc_inverse_filter(audio, sr, order=24, blend=0.70):
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


def _remove_vocal_formants(spectral_env, lifter_bins=60):
    log_env = np.log(np.maximum(spectral_env, 1e-8))
    cep     = np.fft.irfft(log_env, axis=1)
    cep_l   = cep.copy()
    cep_l[:, 1:lifter_bins] = 0.0
    cep_l[:, -lifter_bins:] = 0.0
    log_flat = np.fft.rfft(cep_l, axis=1).real
    return np.exp(log_flat).astype(np.float32)


def _extract_body_lpc(ref_path, sr, order=20):
    import librosa
    ref, ref_sr = sf.read(ref_path)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if ref_sr != sr:
        ref = librosa.resample(ref.astype(np.float32), orig_sr=ref_sr, target_sr=sr)
    mid = len(ref) // 3
    seg = ref[mid:2*mid].astype(np.float64)
    seg /= (np.max(np.abs(seg)) + 1e-8)
    r = np.correlate(seg, seg, mode='full')
    r = r[len(r)//2 : len(r)//2 + order + 1]
    r[0] += 1e-6
    try:
        a = solve_toeplitz(r[:order], r[1:order+1])
        poles = np.roots(np.concatenate([[1.0], -a]))
        if np.any(np.abs(poles) >= 0.999):
            a *= 0.95
        return a.astype(np.float32)
    except Exception as e:
        print(f"[Rebridge] Body LPC failed ({e})")
        return np.zeros(order, dtype=np.float32)


def run_rebridge(instrument, config_path='config.yaml', outdir='soniq_outputs'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vocals_path   = os.path.join(outdir, 'conditioned_vocals.wav')
    seedvc_path   = os.path.join(outdir, 'seed_vc_output.wav')
    l2_path       = os.path.join(outdir, 'extracted_params.npz')
    ddsp_out      = os.path.join(outdir, 'ddsp_params.npz')

    for p in [vocals_path, seedvc_path, l2_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[Rebridge] Missing required file: {p}")

    print(f"\n[Rebridge] Re-generating ddsp_params.npz for '{instrument}'")
    print(f"[Rebridge] Vocals  : {vocals_path}")
    print(f"[Rebridge] Seed-VC : {seedvc_path}")

    # Load Seed-VC output for timbral analysis
    analysis_audio, analysis_sr = sf.read(seedvc_path)
    if analysis_audio.ndim > 1:
        analysis_audio = analysis_audio.mean(axis=1)

    # Load original vocals for pitch-accurate WORLD F0
    src, src_sr = sf.read(vocals_path)
    if src.ndim > 1:
        src = src.mean(axis=1)
    if src_sr != analysis_sr:
        import librosa as _lb
        src = _lb.resample(src.astype(np.float32), orig_sr=src_sr, target_sr=analysis_sr)

    min_smp        = min(len(src), len(analysis_audio))
    src_f64        = src[:min_smp].astype(np.float64)
    analysis_audio = analysis_audio[:min_smp]

    print("[Rebridge] LPC inverse filter on Seed-VC output ...")
    audio_deformed     = _lpc_inverse_filter(analysis_audio, analysis_sr, order=24, blend=0.70)
    audio_f64          = analysis_audio.astype(np.float64)
    audio_deformed_f64 = audio_deformed.astype(np.float64)

    print("[Rebridge] WORLD F0 (DIO + StoneMask) on original vocals ...")
    f0_world, t_world = pw.dio(src_f64, analysis_sr, frame_period=WORLD_FRAME_PERIOD)
    f0_world          = pw.stonemask(src_f64, f0_world, t_world, analysis_sr)

    print("[Rebridge] Spectral envelope (CheapTrick) + aperiodicity (D4C) ...")
    spectral_env = pw.cheaptrick(audio_deformed_f64, f0_world, t_world,
                                  analysis_sr, fft_size=None)
    aperiodicity = pw.d4c(audio_f64, f0_world, t_world, analysis_sr)

    # ── CREPE + WORLD merge (ROOT CAUSE FIX: fill_value = (0, 0)) ────────────
    print("[Rebridge] CREPE+WORLD F0 merge ...")
    l2       = np.load(l2_path)
    f0_crepe = l2['f0']
    conf     = l2['confidence'] if 'confidence' in l2 else None
    vibrato  = l2['vibrato'] if 'vibrato' in l2 else None

    min_len  = min(len(f0_world), len(f0_crepe))
    f0_c = f0_crepe[:min_len]
    f0_w = f0_world[:min_len]

    if conf is not None:
        c            = conf[:min_len]
        f0_c_filled  = f0_c.copy()
        high_conf_mask = (c >= 0.45) & (f0_c > 30.0)

        if high_conf_mask.sum() > 1:
            hi_idx  = np.where(high_conf_mask)[0]
            fill_fn = i1d(hi_idx, f0_c_filled[hi_idx], kind='linear',
                          bounds_error=False,
                          fill_value=(0.0, 0.0))   # FIX: no stale pitch extrapolation
            lo_idx  = np.where(~high_conf_mask)[0]
            f0_c_filled[lo_idx] = fill_fn(lo_idx)
        else:
            f0_c_filled[:] = 0.0   # no high-confidence frames at all → use WORLD

        voiced_crepe = f0_c_filled > 30.0
        voiced_world = f0_w > 30.0
        f0_primary   = np.where(voiced_crepe, f0_c_filled,
                                np.where(voiced_world, f0_w, 0.0)).astype(np.float32)
        n_crepe = int(voiced_crepe.sum())
        n_world = int((~voiced_crepe & voiced_world).sum())
        print(f"[Rebridge] F0 sources: CREPE={n_crepe} frames  WORLD fallback={n_world} frames")
    else:
        f0_primary = np.where(f0_c > 30.0, f0_c, f0_w).astype(np.float32)

    # Align all arrays to f0_primary length
    spectral_env = spectral_env[:min_len]
    aperiodicity = aperiodicity[:min_len]
    if vibrato is not None:
        vibrato = vibrato[:min_len].astype(np.float32)

    n_frames = len(f0_primary)
    for arr_name, arr in [('spectral_env', spectral_env), ('aperiodicity', aperiodicity)]:
        if arr.shape[0] != n_frames:
            t_o = np.linspace(0, 1, arr.shape[0])
            t_n = np.linspace(0, 1, n_frames)
            if arr_name == 'spectral_env':
                spectral_env = i1d(t_o, arr, axis=0, kind='linear')(t_n)
            else:
                aperiodicity = i1d(t_o, arr, axis=0, kind='linear')(t_n)

    # Cepstral formant suppression
    print("[Rebridge] Cepstral formant suppression ...")
    try:
        spectral_env_flat = _remove_vocal_formants(spectral_env, lifter_bins=60)
        loudness_flat     = 20.0 * np.log10(
            np.sqrt(np.mean(spectral_env_flat**2, axis=1)) + 1e-8)
    except Exception as e:
        print(f"[Rebridge] Formant suppression failed ({e}) — using raw envelope")
        spectral_env_flat = spectral_env.astype(np.float32)
        loudness_flat     = 20.0 * np.log10(
            np.sqrt(np.mean(spectral_env**2, axis=1)) + 1e-8)

    loudness_orig = 20.0 * np.log10(
        np.sqrt(np.mean(spectral_env**2, axis=1)) + 1e-8)

    # Body resonance LPC
    body_lpc_a = None
    ref_path = cfg.get('instruments', {}).get(instrument, {}).get('reference', '')
    if ref_path and os.path.isfile(ref_path):
        print(f"[Rebridge] Body LPC from: {ref_path}")
        body_lpc_a = _extract_body_lpc(ref_path, analysis_sr, order=20)

    # Save
    save_dict = dict(
        ddsp_f0              = f0_primary.astype(np.float32),
        ddsp_loudness        = loudness_orig.astype(np.float32),
        ddsp_loudness_flat   = loudness_flat.astype(np.float32),
        ddsp_envelope        = spectral_env.astype(np.float32),
        ddsp_envelope_flat   = spectral_env_flat.astype(np.float32),
        ddsp_aperiodicity    = aperiodicity.astype(np.float32),
        ddsp_frame_period_ms = np.array([WORLD_FRAME_PERIOD], dtype=np.float32),
    )
    if vibrato    is not None: save_dict['ddsp_vibrato'] = vibrato
    if body_lpc_a is not None: save_dict['body_lpc_a']   = body_lpc_a

    np.savez(ddsp_out, **save_dict)
    voiced_count = int((f0_primary > 30.0).sum())
    print(f"[Rebridge] ✅ Saved: {ddsp_out}  frames={n_frames}  voiced={voiced_count}")
    return ddsp_out


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'piano'
    run_rebridge(inst)
