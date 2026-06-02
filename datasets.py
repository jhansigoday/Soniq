"""
prepare_datasets.py  —  Dataset Download & Preprocessing for Seed-VC Training

Downloads and prepares training data for piano, flute, and guitar instruments.

DATASETS:
  Piano  → MAESTRO v3.0.0       (Yamaha Disklavier piano recordings, ~200 hr)
  Flute  → URMP (flute stems)   (University of Rochester Multi-Modal Music, clean stems)
  Guitar → NSynth (guitar set)  (Google Magenta, 305,979 instrument clips)

USAGE:
  # Prepare all three instruments
  python prepare_datasets.py --instruments piano flute guitar

  # Prepare only piano
  python prepare_datasets.py --instruments piano

  # Use a custom base directory
  python prepare_datasets.py --instruments piano --base_dir /data/soniq

OUTPUT STRUCTURE:
  instruments/
    piano/
      train/           ← ~80% of clips, 3–30 s each
      reference.wav    ← 30 s high-quality reference for Seed-VC
    flute/
      train/
      reference.wav
    guitar/
      train/
      reference.wav

DISK SPACE REQUIREMENTS:
  MAESTRO v3 (piano):  ~30 GB download, ~5 GB after processing
  URMP (flute):        ~3 GB download,  ~400 MB after processing
  NSynth guitar:       ~22 GB download, ~800 MB after processing

NOTE: NSynth and MAESTRO are large. The script downloads only the minimum
      subset needed for training. Use --max_minutes to limit dataset size.
"""

import os
import sys
import argparse
import json
import shutil
import urllib.request
import zipfile
import tarfile
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


TARGET_SR   = 44100
TARGET_LUFS = -14.0
MIN_CLIP_S  = 2.0
MAX_CLIP_S  = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_clip(src_path: str, dst_path: str,
                     min_s: float = MIN_CLIP_S,
                     max_s: float = MAX_CLIP_S) -> bool:
    """
    Normalise, trim silence, and save at 44.1 kHz 24-bit.
    Returns True if clip was saved, False if skipped (too short, too noisy).
    """
    try:
        audio, sr = sf.read(src_path)
    except Exception:
        return False

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    dur = len(trimmed) / sr

    if dur < min_s or dur > max_s:
        return False

    # Check RMS — skip clips that are pure silence
    rms = np.sqrt(np.mean(trimmed**2))
    if rms < 1e-4:
        return False

    # Loudness normalise to -14 LUFS
    meter   = pyln.Meter(sr)
    inp     = trimmed[:, np.newaxis]
    current = meter.integrated_loudness(inp)
    if not (np.isinf(current) or np.isnan(current)):
        trimmed = pyln.normalize.loudness(inp, current, TARGET_LUFS)[:, 0]

    peak = np.max(np.abs(trimmed))
    if peak > 0.99:
        trimmed = trimmed / peak * 0.99

    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    sf.write(dst_path, trimmed.astype(np.float32), sr, subtype='PCM_24')
    return True


def build_reference(clips_dir: str, out_path: str, target_dur_s: float = 30.0):
    """
    Concatenate diverse clips into a single reference WAV (~30 s).
    Used by Seed-VC's reference encoder.
    """
    clips = sorted(Path(clips_dir).glob('*.wav'))
    random.shuffle(clips)

    segments = []
    total    = 0.0
    for p in clips:
        audio, sr = sf.read(str(p))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segments.append(audio)
        total += len(audio) / sr
        if total >= target_dur_s:
            break

    if not segments:
        print(f"  ⚠️  No clips found in {clips_dir} — reference not built")
        return

    # Crossfade 50 ms between segments
    fade_s = int(TARGET_SR * 0.05)
    result = segments[0]
    for seg in segments[1:]:
        if len(result) < fade_s or len(seg) < fade_s:
            result = np.concatenate([result, seg])
            continue
        fade_out = np.linspace(1, 0, fade_s)
        fade_in  = np.linspace(0, 1, fade_s)
        result[-fade_s:] = result[-fade_s:] * fade_out + seg[:fade_s] * fade_in
        result = np.concatenate([result, seg[fade_s:]])
        if len(result) / TARGET_SR >= target_dur_s:
            break

    result = result[:int(target_dur_s * TARGET_SR)]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sf.write(out_path, result.astype(np.float32), TARGET_SR, subtype='PCM_24')
    print(f"  ✅ Reference saved: {out_path}  ({len(result)/TARGET_SR:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Piano: MAESTRO v3
# ─────────────────────────────────────────────────────────────────────────────

MAESTRO_JSON_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0.json"
)
MAESTRO_BASE_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
)

def prepare_piano(base_dir: str, max_minutes: int = 30):
    """
    Downloads a representative subset of MAESTRO (piano) for training.
    max_minutes: approximate minutes of audio to download (default 30 min).
    """
    out_train = os.path.join(base_dir, 'piano', 'train')
    out_ref   = os.path.join(base_dir, 'piano', 'reference.wav')
    os.makedirs(out_train, exist_ok=True)

    print("\n[Piano] Downloading MAESTRO metadata ...")
    json_path = os.path.join(base_dir, 'maestro-v3.0.0.json')
    if not os.path.isfile(json_path):
        urllib.request.urlretrieve(MAESTRO_JSON_URL, json_path)

    with open(json_path) as f:
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
    entries.sort(key=lambda x: x['duration'])

    downloaded_s = 0.0
    max_s        = max_minutes * 60.0
    n_ok         = 0

    for i, info in enumerate(entries):
        if downloaded_s >= max_s:
            break
        fname = info.get('audio_filename', '')
        if not fname:
            continue
        url      = MAESTRO_BASE_URL + fname.lstrip('./')
        dst_raw  = os.path.join(base_dir, 'maestro_raw',
                                 os.path.basename(fname))
        dst_clip = os.path.join(out_train, f"piano_{i:04d}.wav")

        if not os.path.isfile(dst_raw):
            print(f"  Downloading {os.path.basename(fname)} ...")
            os.makedirs(os.path.dirname(dst_raw), exist_ok=True)
            try:
                urllib.request.urlretrieve(url, dst_raw)
            except Exception as e:
                print(f"    ⚠️  Download failed: {e}")
                continue

        if preprocess_clip(dst_raw, dst_clip, min_s=3.0, max_s=20.0):
            dur = sf.info(dst_clip).duration
            downloaded_s += dur
            n_ok += 1

        if i % 10 == 0:
            print(f"  Piano clips: {n_ok}  ({downloaded_s/60:.1f} min)")

    print(f"  ✅ Piano: {n_ok} clips  ({downloaded_s/60:.1f} min)")
    build_reference(out_train, out_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Flute: URMP dataset
# ─────────────────────────────────────────────────────────────────────────────

URMP_URL = "https://datashare.ed.ac.uk/download/DS_10283_2950.zip"
# Note: URMP requires registration at Edinburgh DataShare. If the direct link
# fails, download manually from https://datashare.ed.ac.uk/handle/10283/2950
# and place the zip at instruments/urmp_raw/URMP.zip

def prepare_flute(base_dir: str, max_minutes: int = 20):
    out_train = os.path.join(base_dir, 'flute', 'train')
    out_ref   = os.path.join(base_dir, 'flute', 'reference.wav')
    os.makedirs(out_train, exist_ok=True)

    urmp_zip  = os.path.join(base_dir, 'urmp_raw', 'URMP.zip')
    urmp_dir  = os.path.join(base_dir, 'urmp_raw', 'URMP')

    if not os.path.isdir(urmp_dir):
        if not os.path.isfile(urmp_zip):
            print(f"\n[Flute] URMP dataset requires manual download.")
            print(f"  1. Visit https://datashare.ed.ac.uk/handle/10283/2950")
            print(f"  2. Register and download 'DS_10283_2950.zip'")
            print(f"  3. Place it at: {urmp_zip}")
            print(f"  Then re-run this script.")
            return
        print(f"\n[Flute] Extracting URMP ...")
        with zipfile.ZipFile(urmp_zip, 'r') as z:
            z.extractall(os.path.dirname(urmp_zip))

    # Find all flute stem WAV files (named like AuSep_*_fl_*.wav)
    flute_files = sorted(Path(urmp_dir).rglob('AuSep_*_fl_*.wav'))
    if not flute_files:
        # Try alternate naming
        flute_files = sorted(Path(urmp_dir).rglob('*flute*.wav'))

    if not flute_files:
        print(f"  ⚠️  No flute stem files found in {urmp_dir}")
        print(f"       Expected files matching: AuSep_*_fl_*.wav")
        return

    n_ok = 0
    total_s = 0.0
    max_s   = max_minutes * 60.0

    for i, src in enumerate(flute_files):
        if total_s >= max_s:
            break
        dst = os.path.join(out_train, f"flute_{i:04d}.wav")
        if preprocess_clip(str(src), dst, min_s=2.0, max_s=30.0):
            dur = sf.info(dst).duration
            total_s += dur
            n_ok += 1

    print(f"  ✅ Flute: {n_ok} clips  ({total_s/60:.1f} min)")
    build_reference(out_train, out_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Guitar: NSynth
# ─────────────────────────────────────────────────────────────────────────────

NSYNTH_TRAIN_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
)

def prepare_guitar(base_dir: str, max_minutes: int = 20):
    """
    Downloads NSynth train set and extracts guitar clips.
    NSynth clips are 4 s each at 16 kHz (resampled to 44.1 kHz).
    """
    out_train = os.path.join(base_dir, 'guitar', 'train')
    out_ref   = os.path.join(base_dir, 'guitar', 'reference.wav')
    os.makedirs(out_train, exist_ok=True)

    nsynth_tar = os.path.join(base_dir, 'nsynth_raw', 'nsynth-train.tar.gz')
    nsynth_dir = os.path.join(base_dir, 'nsynth_raw', 'nsynth-train')

    if not os.path.isdir(nsynth_dir):
        if not os.path.isfile(nsynth_tar):
            print(f"\n[Guitar] Downloading NSynth train (~22 GB) ...")
            print(f"  This will take a while. Download to: {nsynth_tar}")
            os.makedirs(os.path.dirname(nsynth_tar), exist_ok=True)
            urllib.request.urlretrieve(NSYNTH_TRAIN_URL, nsynth_tar)

        print(f"[Guitar] Extracting NSynth (guitar files only) ...")
        # Extract only guitar files to save disk
        os.makedirs(nsynth_dir, exist_ok=True)
        with tarfile.open(nsynth_tar, 'r:gz') as tar:
            guitar_members = [m for m in tar.getmembers()
                              if 'guitar' in m.name.lower() and m.name.endswith('.wav')]
            tar.extractall(path=nsynth_dir, members=guitar_members[:5000])
            print(f"  Extracted {len(guitar_members)} guitar clips")

    # Process guitar clips
    guitar_files = sorted(Path(nsynth_dir).rglob('guitar_*.wav'))
    if not guitar_files:
        guitar_files = sorted(Path(nsynth_dir).rglob('*guitar*.wav'))

    if not guitar_files:
        print(f"  ⚠️  No guitar files found in {nsynth_dir}")
        return

    n_ok    = 0
    total_s = 0.0
    max_s   = max_minutes * 60.0

    for i, src in enumerate(guitar_files):
        if total_s >= max_s:
            break
        dst = os.path.join(out_train, f"guitar_{i:04d}.wav")
        if preprocess_clip(str(src), dst, min_s=1.5, max_s=10.0):
            dur = sf.info(dst).duration
            total_s += dur
            n_ok += 1

    print(f"  ✅ Guitar: {n_ok} clips  ({total_s/60:.1f} min)")
    build_reference(out_train, out_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_urmp_instrument(base_dir: str, instrument: str,
                              stem_code: str, max_minutes: int = 20):
    """
    Generic URMP extractor for any instrument stem.
    stem_code: URMP abbreviation e.g. 'vn' (violin), 'vc' (cello), 'tpt' (trumpet)
    Reuses the URMP zip already downloaded for flute if present.
    """
    out_train = os.path.join(base_dir, instrument, 'train')
    out_ref   = os.path.join(base_dir, instrument, 'reference.wav')
    os.makedirs(out_train, exist_ok=True)

    urmp_zip  = os.path.join(base_dir, 'urmp_raw', 'URMP.zip')
    urmp_dir  = os.path.join(base_dir, 'urmp_raw', 'URMP')

    if not os.path.isdir(urmp_dir):
        if not os.path.isfile(urmp_zip):
            print(f"\n[{instrument.title()}] URMP dataset not found.")
            print(f"  URMP is shared with flute. If you downloaded it for flute,")
            print(f"  place the zip at: {urmp_zip}")
            print(f"  Or download from: https://datashare.ed.ac.uk/handle/10283/2950")
            print(f"  Then re-run: python datasets.py --instruments {instrument}")
            return
        print(f"\n[{instrument.title()}] Extracting URMP ...")
        with zipfile.ZipFile(urmp_zip, 'r') as z:
            z.extractall(os.path.dirname(urmp_zip))

    # Find stems by URMP abbreviation code
    stem_files = sorted(Path(urmp_dir).rglob(f'AuSep_*_{stem_code}_*.wav'))
    if not stem_files:
        stem_files = sorted(Path(urmp_dir).rglob(f'*{instrument}*.wav'))

    if not stem_files:
        print(f"  ⚠️  No {instrument} stem files found (code='{stem_code}')")
        print(f"       Expected: AuSep_*_{stem_code}_*.wav")
        print(f"       Falling back to synthesised reference ...")
        from reference_generator import ensure_reference
        ensure_reference(instrument, 'config.yaml')
        return

    n_ok    = 0
    total_s = 0.0
    max_s   = max_minutes * 60.0

    for i, src in enumerate(stem_files):
        if total_s >= max_s:
            break
        dst = os.path.join(out_train, f"{instrument}_{i:04d}.wav")
        if preprocess_clip(str(src), dst, min_s=2.0, max_s=30.0):
            dur = sf.info(dst).duration
            total_s += dur
            n_ok += 1

    print(f"  ✅ {instrument.title()}: {n_ok} clips  ({total_s/60:.1f} min)")
    if n_ok > 0:
        build_reference(out_train, out_ref)
    else:
        print(f"  ⚠️  No clips processed — using synthesised reference")
        from reference_generator import ensure_reference
        ensure_reference(instrument, 'config.yaml')


def prepare_violin(base_dir: str, max_minutes: int = 20):
    print(f"\n[Violin] Extracting URMP violin stems (vn) ...")
    _prepare_urmp_instrument(base_dir, 'violin', 'vn', max_minutes)


def prepare_cello(base_dir: str, max_minutes: int = 20):
    print(f"\n[Cello] Extracting URMP cello stems (vc) ...")
    _prepare_urmp_instrument(base_dir, 'cello', 'vc', max_minutes)


def prepare_trumpet(base_dir: str, max_minutes: int = 20):
    print(f"\n[Trumpet] Extracting URMP trumpet stems (tpt) ...")
    _prepare_urmp_instrument(base_dir, 'trumpet', 'tpt', max_minutes)



    parser = argparse.ArgumentParser(
        description='Download and prepare training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--instruments', nargs='+',
                        choices=['piano', 'flute', 'guitar',
                                 'violin', 'cello', 'trumpet'],
                        default=['piano', 'flute', 'guitar'])
    parser.add_argument('--base_dir',    default='instruments',
                        help='Root directory for datasets')
    parser.add_argument('--max_minutes', type=int, default=30,
                        help='Max minutes of audio per instrument (default 30)')
    args = parser.parse_args()

    print(f"\n🎵 Dataset Preparation — SONIQ v5.3")
    print(f"   Instruments : {args.instruments}")
    print(f"   Base dir    : {args.base_dir}")
    print(f"   Max minutes : {args.max_minutes}\n")

    for inst in args.instruments:
        print(f"\n{'='*50}")
        print(f"  Preparing: {inst.upper()}")
        print(f"{'='*50}")
        if inst == 'piano':
            prepare_piano(args.base_dir,   max_minutes=args.max_minutes)
        elif inst == 'flute':
            prepare_flute(args.base_dir,   max_minutes=args.max_minutes)
        elif inst == 'guitar':
            prepare_guitar(args.base_dir,  max_minutes=args.max_minutes)
        elif inst == 'violin':
            prepare_violin(args.base_dir,  max_minutes=args.max_minutes)
        elif inst == 'cello':
            prepare_cello(args.base_dir,   max_minutes=args.max_minutes)
        elif inst == 'trumpet':
            prepare_trumpet(args.base_dir, max_minutes=args.max_minutes)

    print(f"\n{'='*50}")
    print(f"  ✅ Dataset preparation complete")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    for inst in args.instruments:
        print(f"  python train_seed_vc.py --instrument {inst} --epochs 100 --batch 2 --lr 0.00001 --precision 16-mixed")


if __name__ == '__main__':
    main()