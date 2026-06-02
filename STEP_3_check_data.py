"""
STEP3_check_data.py
===================
Run AFTER datasets.py finishes downloading.
Checks that you have enough piano clips to train.

    cd ~/Soniq
    python STEP3_check_data.py
"""

import os
import glob

print("\n============================================")
print("  Checking your training data...")
print("============================================\n")

train_dir = "instruments/piano/train"
ref_file  = "instruments/piano/reference.wav"

# ── Count clips ──────────────────────────────────────────────────────────────
clips = glob.glob(os.path.join(train_dir, "*.wav"))

if not clips:
    print("❌ NO CLIPS FOUND in instruments/piano/train/")
    print()
    print("   This means datasets.py still didn't download correctly.")
    print("   Try running this command to see what error happens:")
    print()
    print("   python datasets.py --instruments piano --max_minutes 30 --base_dir instruments")
    print()
    print("   If you see a download error, your internet connection may be")
    print("   blocking Google Cloud Storage. See the manual download option below.")
    exit(1)

# ── Measure total duration ────────────────────────────────────────────────────
import soundfile as sf

total_minutes = 0.0
bad_clips = 0
for p in clips:
    try:
        info = sf.info(p)
        total_minutes += info.duration / 60.0
    except Exception:
        bad_clips += 1

print(f"  📁 Training clips found : {len(clips)}")
print(f"  ⏱️  Total duration       : {total_minutes:.1f} minutes")
print(f"  🔇 Unreadable clips     : {bad_clips}")
print()

# ── Check reference ──────────────────────────────────────────────────────────
if os.path.isfile(ref_file):
    ref_info = sf.info(ref_file)
    print(f"  🎵 Reference file       : ✅ exists ({ref_info.duration:.1f}s)")
else:
    print(f"  🎵 Reference file       : ❌ MISSING ({ref_file})")

print()

# ── Verdict ──────────────────────────────────────────────────────────────────
if total_minutes < 5:
    print("⚠️  WARNING: Less than 5 minutes of data — training will be poor quality.")
    print("   Run datasets.py again with a higher --max_minutes value.")
    print("   Minimum recommended: 20 minutes.")
elif total_minutes < 15:
    print("⚠️  OK but could be better. 15–30 minutes gives much better results.")
    print("   Consider running datasets.py --max_minutes 30 again.")
else:
    print(f"✅ GOOD! {total_minutes:.1f} minutes of piano data. Ready to train!")
    print()
    print("Next step — start training:")
    print()
    print("  python train_seed_vc.py \\")
    print("      --instrument piano \\")
    print("      --epochs 200 \\")
    print("      --lr 1e-5 \\")
    print("      --batch 4 \\")
    print("      --precision 16-mixed")
    print()
    print("Training will take 4–6 hours. You can let it run overnight.")
    print("When it finishes, run:  python STEP4_run_pipeline.py  your_voice.wav")
