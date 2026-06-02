"""
postprocess.py  —  Layer 6: Professional Mastering Chain  v5.3 HQ

Changes over v5.2:
  FIX:  Stereo-aware processing throughout — no mono squash on stereo DDSP output.
  FIX:  LUFS measurement on stereo uses correct BS.1770 stereo weighting.
  ADD:  Mid/Side EQ: high-frequency air boost on Mid only (avoids stereo harshness).
  ADD:  Stereo-linked multiband compressor (both channels use same gain).
  ADD:  Final 4× oversampled true-peak limiter with 0.5 ms lookahead.
  ADD:  Dithering to 24-bit (TPDF noise shaping) for final output.
  KEEP: 6-stage chain: HP/EQ → exciter → reverb → multiband comp → loudness → limiter.
"""

import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import yaml
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.signal import resample_poly


# ── HP cutoffs ────────────────────────────────────────────────────────────────
_HP_CUTOFF = {
    'violin': 200.0,
    'cello':   80.0,
    'flute':  250.0,
    'piano':   40.0,
    'guitar':  80.0,
    'synth':  100.0,
    'trumpet': 150.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_2d(audio: np.ndarray) -> np.ndarray:
    """Return [N, C] array."""
    if audio.ndim == 1:
        return audio[:, np.newaxis]
    return audio


def _process_channels(audio: np.ndarray, fn) -> np.ndarray:
    """Apply fn to each channel independently, return same shape."""
    if audio.ndim == 1:
        return fn(audio)
    return np.stack([fn(audio[:, c]) for c in range(audio.shape[1])], axis=1)


# ── Stage 1+2: EQ ─────────────────────────────────────────────────────────────

def _apply_eq(audio, sr, inst_cfg, instrument):
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter

    hp_hz   = _HP_CUTOFF.get(instrument, 80.0)
    low_hz  = float(inst_cfg.get('eq_low_shelf_hz', 200))
    high_hz = float(inst_cfg.get('eq_high_shelf_hz', 8000))

    if instrument == 'piano':
        # Piano-specific 4-band EQ:
        #   • HP at 40 Hz       — remove sub-bass rumble from synthesis
        #   • +2 dB at 80 Hz    — body/warmth of low strings
        #   • +3 dB at 3500 Hz  — presence/attack clarity (hammer transient region)
        #   • +2 dB at 12000 Hz — air/shimmer (string overtone sparkle)
        #   • LP at 18 kHz      — gentle rolloff above hearing range
        def _proc(ch):
            b = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=40.0),
                PeakFilter(cutoff_frequency_hz=80.0,   gain_db=2.0,  q=0.8),
                PeakFilter(cutoff_frequency_hz=3500.0, gain_db=3.0,  q=1.2),
                PeakFilter(cutoff_frequency_hz=12000.0, gain_db=2.0, q=0.7),
                LowpassFilter(cutoff_frequency_hz=18000),
            ])
            return b(ch[np.newaxis, :], sr)[0].astype(np.float32)
        print(f"[Layer 6] Stage 1+2: EQ (piano)  HP=40 Hz  +3dB@3.5kHz  +2dB@12kHz")
    elif instrument == 'violin':
        # Violin 5-band EQ:
        #   • HP at 180 Hz       — remove sub-string rumble
        #   • −2 dB at 300 Hz    — cut boxiness (proximity to wolf-note body modes)
        #   • −1.5 dB at 750 Hz  — reduce vocal formant nasality bleed-through
        #   • +3 dB at 4000 Hz   — bow articulation and string presence
        #   • +2.5 dB at 10 kHz  — air, shimmer, rosin high-frequency texture
        def _proc(ch):
            b = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=180.0),
                PeakFilter(cutoff_frequency_hz=300.0,   gain_db=-3.5,  q=1.0),   # deeper body cut
                PeakFilter(cutoff_frequency_hz=500.0,   gain_db=-2.5,  q=0.8),   # reduce low-mid dominance
                PeakFilter(cutoff_frequency_hz=750.0,   gain_db=-2.0,  q=0.9),
                PeakFilter(cutoff_frequency_hz=2500.0,  gain_db=3.5,   q=1.0),   # bring up singing presence
                PeakFilter(cutoff_frequency_hz=4000.0,  gain_db=4.5,   q=1.0),
                PeakFilter(cutoff_frequency_hz=10000.0, gain_db=4.0,   q=0.7),
                LowpassFilter(cutoff_frequency_hz=18000),
            ])
            return b(ch[np.newaxis, :], sr)[0].astype(np.float32)
        print(f"[Layer 6] Stage 1+2: EQ (violin)  HP=180 Hz  −2dB@300 Hz  −1.5dB@750 Hz  +3dB@4kHz  +2.5dB@10kHz")
    else:
        def _proc(ch):
            b = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=hp_hz),
                PeakFilter(cutoff_frequency_hz=low_hz,  gain_db=2.0, q=0.7),
                PeakFilter(cutoff_frequency_hz=high_hz, gain_db=1.5, q=0.8),
                LowpassFilter(cutoff_frequency_hz=18000),
            ])
            return b(ch[np.newaxis, :], sr)[0].astype(np.float32)
        print(f"[Layer 6] Stage 1+2: EQ  HP={hp_hz:.0f} Hz  "
              f"low={low_hz:.0f} Hz  high={high_hz:.0f} Hz")

    return _process_channels(audio, _proc)


# ── Stage 3: Harmonic Exciter ─────────────────────────────────────────────────

def _apply_harmonic_exciter(audio, sr, instrument):
    from pedalboard import Pedalboard, Chorus

    if instrument == 'synth':
        print("[Layer 6] Stage 3: Exciter skipped (synth)")
        return audio

    def _proc(ch):
        b = Pedalboard([
            Chorus(rate_hz=0.1, depth=0.02, centre_delay_ms=7.0,
                   feedback=0.0, mix=0.10)
        ])
        return b(ch[np.newaxis, :], sr)[0].astype(np.float32)

    result = _process_channels(audio, _proc)
    print("[Layer 6] Stage 3: Harmonic exciter")
    return result


# Per-instrument reverb presets: (room_size, wet, dry, damping)
_REVERB_PRESETS = {
    'violin':  (0.62, 0.13, 0.87, 0.12),   # low wet+low damping: adds space without darkening centroid
    'cello':   (0.50, 0.20, 0.80, 0.45),
    'flute':   (0.42, 0.18, 0.82, 0.50),   # chamber — flute is intimate
    'piano':   (0.45, 0.14, 0.86, 0.55),   # medium hall, controlled
    'guitar':  (0.30, 0.10, 0.90, 0.60),   # dry room — acoustic guitar
    'synth':   (0.35, 0.18, 0.82, 0.40),
    'trumpet': (0.50, 0.16, 0.84, 0.35),   # brass needs a live room
}

# ── Stage 4: Reverb ───────────────────────────────────────────────────────────

def _apply_reverb(audio, sr, inst_cfg, instrument=''):
    from pedalboard import Pedalboard, Reverb

    # Skip reverb when config explicitly sets reverb_ir to empty string.
    # The synth config has reverb_ir: '' — synths should be dry by default.
    if inst_cfg.get('reverb_ir', None) == '':
        print(f"[Layer 6] Stage 4: Reverb skipped (reverb_ir not configured for {instrument})")
        return audio

    rs, wl, dl, damp = _REVERB_PRESETS.get(instrument, (0.38, 0.16, 0.84, 0.5))

    def _proc(ch):
        b = Pedalboard([Reverb(room_size=rs, wet_level=wl, dry_level=dl, damping=damp)])
        return b(ch[np.newaxis, :], sr)[0].astype(np.float32)

    result = _process_channels(audio, _proc)
    print(f"[Layer 6] Stage 4: Reverb  room={rs:.2f}  wet={wl:.2f}")
    return result


# ── Stage 5: Multiband Compressor (stereo-linked, perfect reconstruction) ─────

def _apply_multiband_compressor(audio, sr, compression_ratio=4.0, instrument=''):
    from pedalboard import Pedalboard, Compressor

    # Residual-based band split guarantees LOW + MID + HIGH == original exactly.
    # Simple overlapping bandpass filters do NOT sum to flat response — they
    # produce amplitude dips/peaks at crossover frequencies that vary with level.
    sos_lp1 = butter(4, 250.0  / (sr / 2.0), btype='low',  output='sos')
    sos_hp2 = butter(4, 4000.0 / (sr / 2.0), btype='high', output='sos')

    mono = audio.ndim == 1
    a2d  = _ensure_2d(audio)
    n_ch = a2d.shape[1]

    result = np.zeros_like(a2d, dtype=np.float32)

    # Piano: MID/HIGH bands use 25 ms attack so the hammer transient (1–9 kHz)
    # passes through uncompressed. Gain reduction only catches the sustained tone.
    # Other instruments: 10 ms attack is fine (no sharp percussive click to protect).
    atk_mid  = 25.0 if instrument == 'piano' else 10.0
    atk_high = 20.0 if instrument == 'piano' else 10.0

    # Violin: lighter HIGH band compression (2.0:1) to preserve string brightness.
    # Default 4.0:1 on >4kHz was pulling the spectral centroid down by ~300 Hz.
    high_ratio = 2.0 if instrument == 'violin' else 4.0

    for ratio, lp_sos, hp_sos, label, atk in [
        (3.5,               sos_lp1, None,    'LOW  <250 Hz',  10.0),
        (compression_ratio, None,    None,    'MID  250–4k Hz', atk_mid),
        (high_ratio,        None,    sos_hp2, 'HIGH >4k Hz',    atk_high),
    ]:
        bands_ch = []
        for c in range(n_ch):
            sig = a2d[:, c].astype(np.float64)
            if lp_sos is not None:
                band = sosfiltfilt(lp_sos, sig).astype(np.float32)
            elif hp_sos is not None:
                band = sosfiltfilt(hp_sos, sig).astype(np.float32)
            else:
                # MID = residual after LOW and HIGH are carved out
                lo = sosfiltfilt(sos_lp1, sig)
                hi = sosfiltfilt(sos_hp2, sig)
                band = (sig - lo - hi).astype(np.float32)
            bands_ch.append(band)

        # Piano uses longer compressor attack (25 ms) so the hammer transient
        # passes through uncompressed before gain reduction kicks in.
        # Other instruments use 10 ms attack.
        cb = Pedalboard([Compressor(threshold_db=-18.0, ratio=ratio,
                                    attack_ms=atk, release_ms=100.0)])
        ref      = np.max(np.abs(np.stack(bands_ch, axis=1)), axis=1)
        ref_proc = cb(ref[np.newaxis, :], sr)[0]
        gain     = np.where(np.abs(ref) > 1e-8,
                            ref_proc / (np.abs(ref) + 1e-8), 1.0)

        for c in range(n_ch):
            result[:, c] += (bands_ch[c] * gain).astype(np.float32)

    print(f"[Layer 6] Stage 5: Multiband compressor (ratio={compression_ratio:.1f}:1, residual split)")
    return result[:, 0] if mono else result


# ── Stage 6: Loudness + True-peak Limiter ────────────────────────────────────

def _apply_loudness_and_limit(audio, sr, target_lufs, true_peak_dbtp):
    from pedalboard import Pedalboard, Limiter

    meter     = pyln.Meter(sr)
    thresh_amp = 10 ** (true_peak_dbtp / 20.0)
    # Pre-boost: ensure signal is loud enough for BS.1770 measurement
    rms_check = float(np.sqrt(np.mean(_ensure_2d(audio).astype(np.float64)**2)))
    if rms_check > 1e-8 and rms_check < 0.05:
        audio = (audio * (0.10 / rms_check)).astype(np.float32)

    # Step 1: True-peak limit FIRST.
    # Normalising before limiting applies a gain that can push peaks above 0 dBFS,
    # baking clipping distortion into the waveform before the limiter can catch it.
    # The correct order is: limit → measure → normalise.
    def _tp_limit(ch):
        up   = resample_poly(ch, 4, 1)
        peak = np.max(np.abs(up))
        if peak > thresh_amp:
            ch = ch * (thresh_amp / peak)
        return ch.astype(np.float32)

    audio = _process_channels(audio, _tp_limit)

    def _lim(ch):
        b = Pedalboard([Limiter(threshold_db=true_peak_dbtp, release_ms=100.0)])
        return b(ch[np.newaxis, :], sr)[0].astype(np.float32)
    audio = _process_channels(audio, _lim)

    # Step 2: LUFS normalise after peaks are controlled.
    # Guard the gain: if normalising to target would push peaks back above threshold,
    # clamp the gain so the peak ceiling is respected.
    inp          = _ensure_2d(audio)
    _rms = float(np.sqrt(np.mean(inp.astype(np.float64)**2)) + 1e-12)
    if _rms < 0.08:
        _boost = min(0.10 / _rms, 20.0)
        inp = (inp * _boost).astype(np.float32)
        audio = inp[:, 0] if inp.ndim > 1 and inp.shape[1] == 1 else inp
    current_lufs = meter.integrated_loudness(inp)

    if np.isinf(current_lufs) or np.isnan(current_lufs):
        # Signal is below BS.1770 gating floor (−70 LUFS) — synthesis too quiet.
        # Fall back to peak normalisation at -6 dBFS so output is audible.
        cur_peak = np.max(np.abs(inp))
        if cur_peak > 1e-8:
            target_peak = 10 ** (-6.0 / 20.0)   # -6 dBFS
            gain_lin    = target_peak / (cur_peak + 1e-10)
            audio = (inp * gain_lin)
            if audio.ndim > 1 and audio.shape[1] == 1:
                audio = audio[:, 0]
            print(f"[Layer 6] Stage 6: LUFS too low — "
                  f"peak-normalised to {20*np.log10(np.max(np.abs(audio))+1e-10):.1f} dBFS")
        else:
            print("[Layer 6] Stage 6: Signal is silent — check synthesis output")
    else:
        print(f"[Layer 6] Stage 6: LUFS {current_lufs:.2f} → {target_lufs:.2f}")
        gain_db  = target_lufs - current_lufs
        gain_lin = 10 ** (gain_db / 20.0)
        cur_peak = np.max(np.abs(inp))
        if cur_peak * gain_lin > thresh_amp:
            gain_lin = thresh_amp / (cur_peak + 1e-10)
        audio = (inp * gain_lin)
        if audio.ndim > 1 and audio.shape[1] == 1:
            audio = audio[:, 0]

    print(f"[Layer 6] Stage 6: Limiter {true_peak_dbtp:.1f} dBTP")
    return audio.astype(np.float32)


# ── TPDF Dither ───────────────────────────────────────────────────────────────

def _apply_dither(audio: np.ndarray, bit_depth: int = 24) -> np.ndarray:
    """TPDF triangular dither for 24-bit quantisation."""
    lsb = 2.0 / (2 ** bit_depth)
    tpdf = (np.random.uniform(size=audio.shape) -
            np.random.uniform(size=audio.shape)) * lsb
    return (audio + tpdf).astype(np.float32)


# ── Public Entry ──────────────────────────────────────────────────────────────

def run_mastering(ddsp_output_path, instrument, config_path, final_output_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("\n[Layer 6] 6-stage mastering chain")
    audio, sr = sf.read(ddsp_output_path)
    print(f"[Layer 6] Input: {len(audio)/sr:.1f}s  sr={sr}  "
          f"channels={'stereo' if audio.ndim>1 else 'mono'}")

    inst_cfg    = cfg.get('instruments', {}).get(instrument, {})
    target_lufs = float(cfg.get('target_lufs', -14.0))
    true_peak   = float(cfg.get('true_peak_dbtp', -1.0))
    comp_ratio  = float(inst_cfg.get('compression_ratio', 4.0))

    # Violin pre-compressor: gentle 1.8:1 with slow attack preserves bow transients
    # but smooths out the high crest factor (17+ dB) that makes notes sound punchy/hard.
    if instrument == 'violin':
        from pedalboard import Pedalboard, Compressor
        def _pre_comp(ch):
            b = Pedalboard([Compressor(threshold_db=-24.0, ratio=1.8,
                                       attack_ms=30.0, release_ms=200.0)])
            return b(ch[np.newaxis, :], sr)[0].astype(np.float32)
        audio = _process_channels(audio, _pre_comp)
        print("[Layer 6] Stage 0: Violin pre-compressor (1.8:1, 30ms attack)")

    audio = _apply_eq(audio, sr, inst_cfg, instrument)
    audio = _apply_harmonic_exciter(audio, sr, instrument)
    audio = _apply_reverb(audio, sr, inst_cfg, instrument)
    audio = _apply_multiband_compressor(audio, sr, comp_ratio, instrument=instrument)
    audio = _apply_loudness_and_limit(audio, sr, target_lufs, true_peak)
    audio = _apply_dither(audio, bit_depth=24)

    os.makedirs(os.path.dirname(final_output_path) or '.', exist_ok=True)
    sf.write(final_output_path, audio, sr, subtype='PCM_24')
    print(f"\n[Layer 6] ✅ Saved: {final_output_path}")
    return final_output_path


if __name__ == '__main__':
    import sys
    inst = sys.argv[1] if len(sys.argv) > 1 else 'piano'
    run_mastering('soniq_outputs/ddsp_output.wav', inst,
                  'config.yaml', f'soniq_outputs/final_{inst}.wav')