"""
STEP4_run_pipeline.py
=====================
Run this AFTER training is complete.
Usage:
    cd ~/Soniq
    python STEP4_run_pipeline.py  your_voice.wav  piano

Replace "your_voice.wav" with the actual path to your voice recording.
"""

import sys
import os

print("\n============================================")
print("  Pre-flight checks before running pipeline")
print("============================================\n")

# ── Check input file ────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage:  python STEP4_run_pipeline.py  your_voice.wav  piano")
    print()
    print("Supported audio formats: wav, mp3, flac, m4a, ogg")
    exit(1)

input_file = sys.argv[1]
instrument = sys.argv[2] if len(sys.argv) > 2 else 'piano'

if not os.path.isfile(input_file):
    print(f"❌ File not found: {input_file}")
    print()
    print("Make sure the file path is correct.")
    print("Example:  python STEP4_run_pipeline.py recordings/my_voice.wav piano")
    exit(1)

# ── Check checkpoint ─────────────────────────────────────────────────────────
ckpt_options = [
    f"checkpoints/seed_vc_{instrument}.pt",
    f"checkpoints/seedvc_{instrument}.ckpt",
]
ckpt_found = None
for p in ckpt_options:
    if os.path.isfile(p):
        ckpt_found = p
        break

if not ckpt_found:
    print(f"❌ No trained checkpoint found for '{instrument}'")
    print()
    print("Expected one of:")
    for p in ckpt_options:
        print(f"   {p}")
    print()
    print("You need to train first:")
    print()
    print("  python train_seed_vc.py \\")
    print(f"      --instrument {instrument} \\")
    print("      --epochs 200 --lr 1e-5 --batch 4 --precision 16-mixed")
    exit(1)

print(f"  ✅ Input file  : {input_file}")
print(f"  ✅ Instrument  : {instrument}")
print(f"  ✅ Checkpoint  : {ckpt_found}")
print()
print("  Starting pipeline now...")
print("  (This will take 4–10 minutes on GPU)")
print()

# ── Run ──────────────────────────────────────────────────────────────────────
import subprocess

result = subprocess.run(
    ['python', 'orchestrator.py', input_file, instrument],
    check=False
)

if result.returncode == 0:
    print(f"\n✅ Done! Output saved to:  soniq_outputs/final_{instrument}.wav")
else:
    print(f"\n❌ Pipeline failed with error code {result.returncode}")
    print("   Check the error message above.")
    print()
    print("   Common fix: run layers one at a time to find where it breaks:")
    print()
    print(f"   python test_layers_1_3.py --input {input_file}")
    print(f"   python test_layers_4_6.py --instrument {instrument}")
