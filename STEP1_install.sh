#!/bin/bash
# ============================================================
# STEP 1 — Install all missing packages
# Run this FIRST before anything else
# ============================================================

echo ""
echo "============================================"
echo "  Installing missing Python packages..."
echo "============================================"
echo ""

# Make sure you're in the soniq conda env before running this
# (your terminal should show (soniq) on the left)

pip install pytorch-lightning --quiet
pip install soundfile librosa pyloudnorm noisereduce scipy numpy --quiet
pip install pedalboard --quiet
pip install transformers --quiet
pip install pydub --quiet
pip install crepe --quiet
pip install pyworld --quiet
pip install torchaudio --quiet

echo ""
echo "✅ All packages installed!"
echo ""
echo "Now run:  python STEP2_fix_code.py"
