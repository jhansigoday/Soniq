"""
test_layers_4_6.py
──────────────────
Sequential test runner for SONIQ pipeline Layers 4–6.

Usage
-----
  # Run all 3 layers (requires Layers 1–3 outputs to exist)
  python test_layers_4_6.py --instrument violin

  # Run a specific layer only
  python test_layers_4_6.py --instrument violin --layer 4
  python test_layers_4_6.py --instrument violin --layer 5
  python test_layers_4_6.py --instrument violin --layer 6

  # Skip Layer 4 if seed_vc_output.wav + ddsp_params.npz already exist
  python test_layers_4_6.py --instrument violin --skip-layer4

  # Custom output directory
  python test_layers_4_6.py --instrument violin --outdir my_outputs
"""

import argparse
import os
import sys
import time
import traceback
import yaml


# ── Paths ─────────────────────────────────────────────────────────────────────
def make_paths(outdir: str, instrument: str) -> dict:
    return {
        'vocals':      os.path.join(outdir, 'conditioned_vocals.wav'),
        'params':      os.path.join(outdir, 'extracted_params.npz'),
        'embeddings':  os.path.join(outdir, 'hubert_embeddings.npy'),
        'seed_vc':     os.path.join(outdir, 'seed_vc_output.wav'),
        'ddsp_params': os.path.join(outdir, 'ddsp_params.npz'),
        'ddsp_out':    os.path.join(outdir, 'ddsp_output.wav'),
        'final':       os.path.join(outdir, f'final_{instrument}.wav'),
    }


def _require_file(path: str, description: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file missing: {path}\n"
            f"  ({description})\n"
            f"  Run the preceding layers first."
        )


# ── Layer runners ─────────────────────────────────────────────────────────────
def run_layer4(paths: dict, instrument: str, config_path: str) -> float:
    print("\n" + "=" * 60)
    print("  LAYER 4 — Seed-VC Conversion + WORLD Re-Analysis Bridge")
    print("=" * 60)
    _require_file(paths['vocals'],     "Vocals WAV from Layer 1")
    _require_file(paths['embeddings'], "HuBERT embeddings from Layer 3")

    from converter import run_conversion
    t0 = time.time()
    run_conversion(
        source_audio_path=paths['vocals'],      # ← Layer 1 vocals
        instrument=instrument,
        config_path=config_path,
        output_wav_path=paths['seed_vc'],
        ddsp_params_path=paths['ddsp_params'],
        hubert_embeddings_path=paths['embeddings'],
        extracted_params_path=paths['params'],
    )
    elapsed = time.time() - t0
    print(f"\n[Test] ✅ Layer 4 done in {elapsed:.1f}s")
    print(f"         Seed-VC WAV : {paths['seed_vc']}")
    print(f"         DDSP params : {paths['ddsp_params']}")

    from quality_check_additions import check_layer4
    qc = check_layer4(paths['seed_vc'], paths['params'])
    print(f"[Test] QC Layer 4: {qc.upper()}")
    return elapsed


def run_layer5(paths: dict, instrument: str, config_path: str) -> float:
    print("\n" + "=" * 60)
    print("  LAYER 5 — DDSP Physical Instrument Synthesis")
    print("=" * 60)
    _require_file(paths['ddsp_params'], "DDSP params from Layer 4 bridge")

    from synthesiser import run_synthesis
    t0 = time.time()
    run_synthesis(
        ddsp_params_path=paths['ddsp_params'],
        instrument=instrument,
        config_path=config_path,
        output_path=paths['ddsp_out'],
    )
    elapsed = time.time() - t0
    print(f"\n[Test] ✅ Layer 5 done in {elapsed:.1f}s  →  {paths['ddsp_out']}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ref = cfg.get('instruments', {}).get(instrument, {}).get('reference', '')
    if ref and os.path.isfile(ref):
        from quality_check_additions import check_layer5
        qc = check_layer5(paths['ddsp_out'], ref)
        print(f"[Test] QC Layer 5: {qc.upper()}")
    else:
        print(f"[Test] QC Layer 5: SKIPPED (reference audio not found: {ref})")
    return elapsed


def run_layer6(paths: dict, instrument: str, config_path: str) -> float:
    print("\n" + "=" * 60)
    print("  LAYER 6 — Professional Mastering Chain (Pedalboard)")
    print("=" * 60)
    _require_file(paths['ddsp_out'], "DDSP synthesis output from Layer 5")

    from postprocess import run_mastering
    t0 = time.time()
    run_mastering(
        ddsp_output_path=paths['ddsp_out'],
        instrument=instrument,
        config_path=config_path,
        final_output_path=paths['final'],
    )
    elapsed = time.time() - t0
    print(f"\n[Test] ✅ Layer 6 done in {elapsed:.1f}s  →  {paths['final']}")

    from quality_check_additions import check_layer6
    qc = check_layer6(paths['final'], config_path)
    print(f"[Test] QC Layer 6: {qc.upper()}")
    return elapsed


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(timings: dict, paths: dict):
    print("\n" + "=" * 60)
    print("  LAYERS 4–6 SUMMARY")
    print("=" * 60)
    total = sum(timings.values())
    for layer, elapsed in timings.items():
        print(f"  {layer}: {elapsed:.1f}s")
    print(f"  ─────────────────────")
    print(f"  Total: {total:.1f}s  ({total/60:.1f} min)")
    print()
    print("  Output files:")
    for name in ('seed_vc', 'ddsp_params', 'ddsp_out', 'final'):
        path   = paths[name]
        exists = "✅" if os.path.isfile(path) else "❌ MISSING"
        size   = f"  ({os.path.getsize(path)/1024:.0f} KB)" if os.path.isfile(path) else ""
        print(f"  {exists}  {path}{size}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test SONIQ pipeline Layers 4–6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--instrument', required=True,
                        choices=['violin', 'piano', 'flute', 'cello', 'guitar', 'synth'],
                        help='Target instrument')
    parser.add_argument('--layer',       type=int,
                        help='Run only this specific layer (4, 5, or 6)')
    parser.add_argument('--skip-layer4', action='store_true',
                        help='Skip Layer 4 — uses existing seed_vc_output.wav + ddsp_params.npz')
    parser.add_argument('--outdir',      default='soniq_outputs',
                        help='Output directory (default: soniq_outputs)')
    parser.add_argument('--config',      default='config.yaml',
                        help='Config file path (default: config.yaml)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    paths   = make_paths(args.outdir, args.instrument)
    timings = {}

    if args.layer:
        layers_to_run = [args.layer]
    elif args.skip_layer4:
        layers_to_run = [5, 6]
    else:
        layers_to_run = [4, 5, 6]

    print(f"\n🎵 SONIQ Layer Test  |  Instrument: {args.instrument}")
    print(f"   Output dir:    {args.outdir}")
    print(f"   Config:        {args.config}")
    print(f"   Layers to run: {layers_to_run}\n")

    for layer_id in layers_to_run:
        try:
            if layer_id == 4:
                timings['L4_seedvc']    = run_layer4(paths, args.instrument, args.config)
            elif layer_id == 5:
                timings['L5_ddsp']      = run_layer5(paths, args.instrument, args.config)
            elif layer_id == 6:
                timings['L6_mastering'] = run_layer6(paths, args.instrument, args.config)
            else:
                print(f"⚠️  Unknown layer {layer_id} — skipping")

        except FileNotFoundError as e:
            print(f"\n❌  DEPENDENCY ERROR in Layer {layer_id}:")
            print(f"   {e}")
            print("\n   Fix: run the preceding layers first, then retry.")
            sys.exit(1)

        except Exception as e:
            print(f"\n❌  ERROR in Layer {layer_id}: {e}")
            traceback.print_exc()
            print("\n   Other completed layers are saved. Fix the error and re-run with --layer flag.")
            sys.exit(1)

    print_summary(timings, paths)


if __name__ == '__main__':
    main()