"""
test_layers_1_3.py
──────────────────
Sequential test runner for SONIQ pipeline Layers 1–3.

Usage
-----
  # Run all 3 layers
  python test_layers_1_3.py --input path/to/audio.wav

  # Run only specific layers (useful for re-running after a crash)
  python test_layers_1_3.py --input path/to/audio.wav --layer 1
  python test_layers_1_3.py --input path/to/audio.wav --layer 2
  python test_layers_1_3.py --input path/to/audio.wav --layer 3

  # Skip Layer 1 (Demucs) if conditioned_vocals.wav already exists
  python test_layers_1_3.py --input path/to/audio.wav --skip-layer1

  # Custom output directory
  python test_layers_1_3.py --input path/to/audio.wav --outdir my_outputs
"""

import argparse
import os
import sys
import time
import traceback


# ── Paths ─────────────────────────────────────────────────────────────────────

def make_paths(outdir: str):
    return {
        'vocals':     os.path.join(outdir, 'conditioned_vocals.wav'),
        'params':     os.path.join(outdir, 'extracted_params.npz'),
        'embeddings': os.path.join(outdir, 'hubert_embeddings.npy'),
    }


# ── Individual layer runners ───────────────────────────────────────────────────

def run_layer1(input_wav: str, paths: dict):
    print("\n" + "="*60)
    print("  LAYER 1 — Voice Ingestion & Source Separation (Demucs)")
    print("="*60)
    from ingestion import run_ingestion
    t0 = time.time()
    run_ingestion(input_wav, paths['vocals'])
    elapsed = time.time() - t0
    print(f"[Test] ✅ Layer 1 done in {elapsed:.1f}s  →  {paths['vocals']}")
    return elapsed


def run_layer2(paths: dict):
    print("\n" + "="*60)
    print("  LAYER 2 — Deep Voice Analysis (CREPE + WORLD)")
    print("="*60)

    vocals = paths['vocals']
    if not os.path.isfile(vocals):
        raise FileNotFoundError(
            f"Layer 2 requires '{vocals}'. Run Layer 1 first, "
            "or use --skip-layer1 only if the file already exists."
        )

    from extractor import run_extraction
    t0 = time.time()
    run_extraction(vocals, paths['params'])
    elapsed = time.time() - t0
    print(f"[Test] ✅ Layer 2 done in {elapsed:.1f}s  →  {paths['params']}")

    # Quick QC
    from quality_check import check_layer2
    qc_result = check_layer2(paths['params'])
    print(f"[Test] QC Layer 2: {qc_result.upper()}")
    return elapsed


def run_layer3(paths: dict):
    print("\n" + "="*60)
    print("  LAYER 3 — HuBERT Voice Encoding")
    print("="*60)

    vocals = paths['vocals']
    if not os.path.isfile(vocals):
        raise FileNotFoundError(
            f"Layer 3 requires '{vocals}'. Run Layer 1 first."
        )

    from encoder import run_encoding
    t0 = time.time()
    run_encoding(vocals, paths['embeddings'])
    elapsed = time.time() - t0
    print(f"[Test] ✅ Layer 3 done in {elapsed:.1f}s  →  {paths['embeddings']}")

    # Quick QC
    from quality_check import check_layer3
    qc_result = check_layer3(paths['embeddings'])
    print(f"[Test] QC Layer 3: {qc_result.upper()}")
    return elapsed


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(timings: dict, paths: dict):
    print("\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    total = sum(timings.values())
    for layer, elapsed in timings.items():
        print(f"  {layer}: {elapsed:.1f}s")
    print(f"  ─────────────────")
    print(f"  Total: {total:.1f}s  ({total/60:.1f} min)")
    print()
    print("  Output files:")
    for name, path in paths.items():
        exists = "✅" if os.path.isfile(path) else "❌ MISSING"
        size   = f"  ({os.path.getsize(path)/1024:.0f} KB)" if os.path.isfile(path) else ""
        print(f"  {exists}  {path}{size}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test SONIQ pipeline Layers 1–3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',       required=True,  help='Path to input audio file (wav/flac/mp3)')
    parser.add_argument('--layer',       type=int,       help='Run only this specific layer (1, 2, or 3)')
    parser.add_argument('--skip-layer1', action='store_true',
                        help='Skip Layer 1 (Demucs) — uses existing conditioned_vocals.wav')
    parser.add_argument('--outdir',      default='soniq_outputs',
                        help='Output directory (default: soniq_outputs)')
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"❌  Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    paths = make_paths(args.outdir)
    timings = {}

    # ── Determine which layers to run ─────────────────────────────────────────
    if args.layer:
        layers_to_run = [args.layer]
    elif args.skip_layer1:
        layers_to_run = [2, 3]
    else:
        layers_to_run = [1, 2, 3]

    print(f"\n🎵 SONIQ Layer Test  |  Input: {args.input}")
    print(f"   Output dir: {args.outdir}")
    print(f"   Layers to run: {layers_to_run}\n")

    # ── Run each layer with error isolation ───────────────────────────────────
    for layer_id in layers_to_run:
        try:
            if layer_id == 1:
                timings['L1_ingestion'] = run_layer1(args.input, paths)
            elif layer_id == 2:
                timings['L2_analysis']  = run_layer2(paths)
            elif layer_id == 3:
                timings['L3_hubert']    = run_layer3(paths)
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