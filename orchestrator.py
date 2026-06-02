"""
orchestrator.py  —  Full ControllaVoice / SONIQ Pipeline (Layers 1–6)

ControllaVoice v5.0 changes:
  • Auto-converts any input format (mp3, flac, ogg, m4a, wav ...) — no manual prep
  • Auto-generates instrument reference.wav if it doesn't exist
  • Passes Layer 2 CREPE F0 params to Layer 4 (primary F0 for DDSP bridge)
  • 6-stage mastering chain in Layer 6

Usage:
    python orchestrator.py <input_audio> <instrument>
    python orchestrator.py my_voice.mp3 violin
    python orchestrator.py recording.m4a piano

Supported input formats: wav, mp3, flac, ogg, m4a, aac, wma, aiff, opus
Instruments: violin | piano | flute | cello | guitar | synth
"""

import os
import sys
import time
import yaml


INSTRUMENTS = ['violin', 'piano', 'flute', 'cello', 'guitar', 'synth', 'trumpet']

SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
                     '.wma', '.aiff', '.aif', '.opus', '.webm']


def pick_instrument() -> str:
    print("\nSelect target instrument:")
    for i, name in enumerate(INSTRUMENTS, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input("\nEnter number (1–6): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(INSTRUMENTS):
            return INSTRUMENTS[int(choice) - 1]
        print("  Invalid choice, try again.")


def run_pipeline(
    input_audio: str,
    instrument: str,
    config_path: str = 'config.yaml',
    outdir: str = 'soniq_outputs',
) -> str:
    """
    Run the full 6-layer ControllaVoice / SONIQ pipeline.
    Returns path to the final output WAV.
    """
    cfg = yaml.safe_load(open(config_path))
    os.makedirs(outdir, exist_ok=True)
    timings = {}

    # ── File paths ────────────────────────────────────────────────────────────
    p = {
        'vocals':      os.path.join(outdir, 'conditioned_vocals.wav'),
        'params':      os.path.join(outdir, 'extracted_params.npz'),
        'embeddings':  os.path.join(outdir, 'hubert_embeddings.npy'),
        'seed_vc':     os.path.join(outdir, 'seed_vc_output.wav'),
        'ddsp_params': os.path.join(outdir, 'ddsp_params.npz'),
        'ddsp_out':    os.path.join(outdir, 'ddsp_output.wav'),
        'final':       os.path.join(outdir, f'final_{instrument}.wav'),
    }

    print(f"\n🎵 ControllaVoice / SONIQ Pipeline  v5.0")
    print(f"   Input:      {input_audio}")
    print(f"   Instrument: {instrument}")
    print(f"   Output dir: {outdir}\n")

    # ── Pre-flight: auto-generate reference audio for selected instrument ─────
    print("=" * 60)
    print("  PRE-FLIGHT — Instrument Reference Check")
    print("=" * 60)
    from reference_generator import ensure_reference
    t = time.time()
    ref_path = ensure_reference(instrument, config_path)
    timings['P0_reference'] = time.time() - t
    print(f"   Reference ready: {ref_path}\n")

    # ── Layer 1: Voice Ingestion & Source Separation ──────────────────────────
    print("=" * 60)
    print("  LAYER 1 — Voice Ingestion & Source Separation (Demucs)")
    print("=" * 60)
    from ingestion import run_ingestion
    from quality_check import check_layer1
    t = time.time()
    run_ingestion(input_audio, p['vocals'])    # auto-converts mp3/flac/etc → wav
    check_layer1(p['vocals'])
    timings['L1_ingestion'] = time.time() - t

    # ── Layer 2: Deep Voice Analysis ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYER 2 — Deep Voice Analysis (CREPE + WORLD)")
    print("=" * 60)
    from extractor import run_extraction
    from quality_check import check_layer2
    t = time.time()
    run_extraction(p['vocals'], p['params'])
    check_layer2(p['params'])
    timings['L2_analysis'] = time.time() - t

    # ── Layer 3: HuBERT Voice Encoding ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYER 3 — HuBERT Voice Encoding (overlapping windows)")
    print("=" * 60)
    from encoder import run_encoding
    from quality_check import check_layer3
    t = time.time()
    run_encoding(p['vocals'], p['embeddings'])
    check_layer3(p['embeddings'])
    timings['L3_hubert'] = time.time() - t

    # ── Layer 4: Seed-VC + WORLD Bridge ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYER 4 — Seed-VC Conversion + WORLD Re-Analysis Bridge")
    print("=" * 60)
    from converter import run_conversion
    from quality_check_additions import check_layer4
    t = time.time()
    run_conversion(
        source_audio_path=p['vocals'],
        instrument=instrument,
        config_path=config_path,
        output_wav_path=p['seed_vc'],
        ddsp_params_path=p['ddsp_params'],
        hubert_embeddings_path=p['embeddings'],
        extracted_params_path=p['params'],       # ← CREPE F0 primary source (v5.0)
    )
    check_layer4(p['seed_vc'], p['params'])
    timings['L4_seedvc'] = time.time() - t

    # ── Layer 5: DDSP Physical Synthesis ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYER 5 — DDSP Physical Instrument Synthesis")
    print("=" * 60)
    from synthesiser import run_synthesis
    from quality_check_additions import check_layer5
    t = time.time()
    run_synthesis(p['ddsp_params'], instrument, config_path, p['ddsp_out'])
    ref_audio = cfg.get('instruments', {}).get(instrument, {}).get('reference', '')
    if ref_audio and os.path.isfile(ref_audio):
        check_layer5(p['ddsp_out'], ref_audio)
    else:
        print(f"[QC] Layer 5: Spectral check skipped (reference not found: {ref_audio})")
    timings['L5_ddsp'] = time.time() - t

    # ── Layer 6: 6-Stage Mastering Chain ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYER 6 — Professional Mastering Chain (6-Stage Pedalboard)")
    print("=" * 60)
    from postprocess import run_mastering
    from quality_check_additions import check_layer6
    t = time.time()
    run_mastering(p['ddsp_out'], instrument, config_path, p['final'])
    check_layer6(p['final'], config_path)
    timings['L6_mastering'] = time.time() - t

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(timings.values())
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  ⏱️  Total: {total/60:.1f} min  ({total:.0f}s)")
    for k, v in timings.items():
        print(f"     {k}: {v:.1f}s")
    print(f"\n  🎵 Final output: {p['final']}")
    print()

    return p['final']


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <input_audio> [instrument]")
        print(f"       Instruments: {' | '.join(INSTRUMENTS)}")
        print(f"       Formats:     {' | '.join(f[1:] for f in SUPPORTED_FORMATS)}")
        sys.exit(1)

    input_file = sys.argv[1]
    ext = os.path.splitext(input_file)[1].lower()

    if not os.path.isfile(input_file):
        print(f"❌  Input file not found: {input_file}")
        sys.exit(1)

    if ext not in SUPPORTED_FORMATS:
        print(f"⚠️  Unrecognised extension '{ext}' — will attempt conversion anyway")

    if len(sys.argv) >= 3:
        chosen_instrument = sys.argv[2].lower()
        if chosen_instrument not in INSTRUMENTS:
            print(f"❌  Unknown instrument '{chosen_instrument}'")
            print(f"   Valid: {', '.join(INSTRUMENTS)}")
            sys.exit(1)
    else:
        chosen_instrument = pick_instrument()

    run_pipeline(input_file, chosen_instrument)