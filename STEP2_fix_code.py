"""
STEP2_fix_code.py
=================
Run this from inside your Soniq folder.
It automatically fixes 3 bugs in your code.

    cd ~/Soniq
    python STEP2_fix_code.py
"""

import os
import re

# ── Helper ─────────────────────────────────────────────────────────────────

def read(path):
    with open(path, 'r') as f:
        return f.read()

def write(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"  ✅ Fixed: {path}")

# ── Check we're in the right folder ────────────────────────────────────────

needed = ['datasets.py', 'config.yaml', 'test_layers_4_6.py']
missing = [f for f in needed if not os.path.isfile(f)]
if missing:
    print(f"\n❌ ERROR: Can't find these files: {missing}")
    print("   Make sure you're running this FROM inside your Soniq folder:")
    print("   cd ~/Soniq")
    print("   python STEP2_fix_code.py")
    exit(1)

print("\n============================================")
print("  Fixing 3 bugs in your code...")
print("============================================\n")


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 1 — datasets.py
# Problem: MAESTRO JSON is in "column" format (each key is a whole column of
#          values), but the code treated it like "row" format.
#          Result: 0 clips downloaded, no training data.
# Fix:    Rewrite the parsing to read the real JSON structure.
# ══════════════════════════════════════════════════════════════════════════════

print("FIX 1/3 — MAESTRO data download (datasets.py)")

OLD_PARSE = '''    with open(json_path) as f:
        meta = json.load(f)

    # Use 'train' split only, sorted by duration, take shortest clips first
    entries = [(k, v) for k, v in meta.items()
               if isinstance(v, dict) and v.get('split') == 'train']
    entries.sort(key=lambda x: x[1].get('duration', 999))'''

NEW_PARSE = '''    with open(json_path) as f:
        meta = json.load(f)

    # MAESTRO v3 JSON is column-oriented:
    #   meta['split']['0'] = 'train', meta['audio_filename']['0'] = '2004/...'
    # Build a list of train-split entries from the column data.
    splits         = meta.get('split', {})
    audio_filenames = meta.get('audio_filename', {})
    durations      = meta.get('duration', {})

    entries = []
    for idx in splits:
        if splits[idx] == 'train':
            entries.append({
                'audio_filename': audio_filenames.get(idx, ''),
                'duration':       float(durations.get(idx, 999)),
            })
    entries.sort(key=lambda x: x['duration'])'''

OLD_LOOP = '''    for i, (uid, info) in enumerate(entries):
        if downloaded_s >= max_s:
            break
        fname = info.get('audio_filename', '')
        if not fname:
            continue
        url      = MAESTRO_BASE_URL + fname.lstrip('./')
        dst_raw  = os.path.join(base_dir, 'maestro_raw',
                                 os.path.basename(fname))
        dst_clip = os.path.join(out_train, f"piano_{i:04d}.wav")'''

NEW_LOOP = '''    for i, info in enumerate(entries):
        if downloaded_s >= max_s:
            break
        fname = info.get('audio_filename', '')
        if not fname:
            continue
        url      = MAESTRO_BASE_URL + fname.lstrip('./')
        dst_raw  = os.path.join(base_dir, 'maestro_raw',
                                 os.path.basename(fname))
        dst_clip = os.path.join(out_train, f"piano_{i:04d}.wav")'''

content = read('datasets.py')

if 'column-oriented' in content:
    print("  ℹ️  Already fixed, skipping")
else:
    if OLD_PARSE not in content:
        print("  ⚠️  Could not find the exact old parse block — printing diff hint:")
        print("     Look for 'isinstance(v, dict)' in datasets.py and replace the")
        print("     prepare_piano() parsing section with the column-oriented version.")
    else:
        content = content.replace(OLD_PARSE, NEW_PARSE)
        content = content.replace(OLD_LOOP, NEW_LOOP)
        write('datasets.py', content)


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 2 — config.yaml
# Problem: ddim_steps is 20 — this makes Seed-VC do only 20 denoising steps,
#          giving blurry, noisy audio.
# Fix:    Set it to 50 for proper quality.
# ══════════════════════════════════════════════════════════════════════════════

print("\nFIX 2/3 — DDIM quality steps (config.yaml)")

content = read('config.yaml')
if 'ddim_steps: 50' in content:
    print("  ℹ️  Already 50 steps, skipping")
elif 'ddim_steps: 20' in content:
    content = content.replace('ddim_steps: 20', 'ddim_steps: 50')
    write('config.yaml', content)
else:
    print("  ⚠️  Could not find 'ddim_steps: 20' — check config.yaml manually")


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 3 — test_layers_4_6.py
# Problem: run_layer4() never passes the CREPE F0 data (extracted_params_path)
#          to run_conversion(). So Layer 4 uses only WORLD F0 — less accurate,
#          causes pitch jumps and discontinuities in the output.
# Fix:    Add extracted_params_path=paths['params'] to the call.
# ══════════════════════════════════════════════════════════════════════════════

print("\nFIX 3/3 — Pass CREPE F0 to Layer 4 (test_layers_4_6.py)")

OLD_CALL = '''    run_conversion(
        source_audio_path=paths['vocals'],      # ← Layer 1 vocals
        instrument=instrument,
        config_path=config_path,
        output_wav_path=paths['seed_vc'],
        ddsp_params_path=paths['ddsp_params'],
        hubert_embeddings_path=paths['embeddings'],
    )'''

NEW_CALL = '''    run_conversion(
        source_audio_path=paths['vocals'],      # ← Layer 1 vocals
        instrument=instrument,
        config_path=config_path,
        output_wav_path=paths['seed_vc'],
        ddsp_params_path=paths['ddsp_params'],
        hubert_embeddings_path=paths['embeddings'],
        extracted_params_path=paths['params'],  # ← CREPE F0 primary source
    )'''

content = read('test_layers_4_6.py')
if "extracted_params_path=paths['params']" in content:
    print("  ℹ️  Already fixed, skipping")
elif OLD_CALL in content:
    content = content.replace(OLD_CALL, NEW_CALL)
    write('test_layers_4_6.py', content)
else:
    print("  ⚠️  Could not find exact call block — fix manually:")
    print("     In run_layer4(), add this line inside run_conversion():")
    print("         extracted_params_path=paths['params'],")


# ── Done ───────────────────────────────────────────────────────────────────

print("\n============================================")
print("  All 3 bugs fixed!")
print("============================================")
print()
print("Next step — download training data:")
print()
print("  python datasets.py --instruments piano --max_minutes 30 --base_dir instruments")
print()
print("(This will take a while to download — leave it running)")
