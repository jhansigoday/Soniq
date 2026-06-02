"""
train_seed_vc.py  —  Seed-VC Fine-Tuning for Instruments  v5.3

Instruments in this phase: piano, flute, guitar
(violin, cello, synth to follow in Phase 2)

DATASET SETUP (do this before training):
  See prepare_datasets.py for automated download and preprocessing.

  Minimum clean audio per instrument:
    piano  → MAESTRO-v3 subset   (≥ 30 min recommended)
    flute  → URMP flute stems     (≥ 20 min recommended)
    guitar → NSynth guitar subset (≥ 20 min recommended)

TRAINING TIME ESTIMATES (NVIDIA A100 / 40 GB):
    piano  → ~4–5 hours  (200 epochs)
    flute  → ~3–4 hours  (200 epochs)
    guitar → ~3–4 hours  (200 epochs)
  On RTX 3090 (24 GB) multiply by ~2×.
  On RTX 4090 (24 GB) similar to A100.

HOW TO RUN:
    python train_seed_vc.py --instrument piano  --epochs 200
    python train_seed_vc.py --instrument flute  --epochs 200
    python train_seed_vc.py --instrument guitar --epochs 200

CHECKPOINTS SAVED TO:
    checkpoints/seedvc_piano.pt
    checkpoints/seedvc_flute.pt
    checkpoints/seedvc_guitar.pt

Copy these to config.yaml checkpoint paths after training:
    checkpoints:
      seed_vc:
        piano:  checkpoints/seed_vc_piano.pt
        flute:  checkpoints/seed_vc_flute.pt
        guitar: checkpoints/seed_vc_guitar.pt
"""

import os
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torchaudio.transforms as T
import sys
sys.path.append("./seed-vc")
from pathlib import Path



# ── Dataset ───────────────────────────────────────────────────────────────────

class InstrumentDataset(Dataset):
    """
    Loads audio clips for fine-tuning Seed-VC.

    Clips are segmented into fixed-length windows with overlap.
    Window length 3 s, hop 1.5 s — gives enough context for Seed-VC
    reference encoder without GPU memory overflow.

    Data augmentation (training only):
      - Random pitch shift ±1 semitone (simulates register variation)
      - Gaussian noise at SNR 40–60 dB (mild robustness)
      - Random gain ±3 dB
    """

    WINDOW_S = 3.0       # seconds per clip
    HOP_S    = 1.5       # hop between clips
    TARGET_SR = 44100

    def __init__(self, folder: str, augment: bool = False):
        self.augment = augment
        self.clips   = []
        self._load_folder(folder)
        if len(self.clips) == 0:
            raise RuntimeError(f"No audio clips found in: {folder}")
        print(f"[Dataset] {len(self.clips)} clips from {folder}")

    def _load_folder(self, folder: str):
        exts = {'.wav', '.mp3', '.flac', '.ogg'}
        win  = int(self.WINDOW_S  * self.TARGET_SR)
        hop  = int(self.HOP_S     * self.TARGET_SR)
        for p in Path(folder).rglob('*'):
            if p.suffix.lower() not in exts:
                continue
            try:
                meta = torchaudio.info(str(p))
                sr   = meta.sample_rate
                dur  = meta.num_frames / sr
            except Exception:
                continue
            if dur < 1.0:
                continue   # skip clips shorter than 1 s

            # Segment the file
            num_clips = max(1, int((dur * self.TARGET_SR - win) / hop) + 1)
            for i in range(num_clips):
                start_s = i * self.HOP_S
                self.clips.append((str(p), sr, start_s))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        path, sr, start_s = self.clips[idx]
        win_s = self.WINDOW_S

        # Load window
        frame_off = int(start_s * sr)
        frame_cnt = int(win_s   * sr)
        try:
            wav, sr_ = torchaudio.load(path,
                                        frame_offset=frame_off,
                                        num_frames=frame_cnt)
        except Exception:
            wav, sr_ = torchaudio.load(path)
            wav = wav[:, frame_off:frame_off + frame_cnt]

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample
        if sr_ != self.TARGET_SR:
            wav = T.Resample(sr_, self.TARGET_SR)(wav)

        # Pad / trim to exact window
        target_len = int(self.WINDOW_S * self.TARGET_SR)
        if wav.shape[-1] < target_len:
            pad = target_len - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, :target_len]

        # Normalise
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.9

        # Augmentation
        if self.augment:
            # Random gain ±3 dB
            gain_db = random.uniform(-3, 3)
            wav = wav * (10 ** (gain_db / 20.0))

            # Random pitch shift ±1 semitone
            if random.random() < 0.5:
                semitones = random.uniform(-1.0, 1.0)
                n_steps   = round(semitones)
                if n_steps != 0:
                    try:
                        wav = T.PitchShift(self.TARGET_SR, n_steps)(wav)
                    except Exception:
                        pass

            # Gaussian noise at SNR ~50 dB
            if random.random() < 0.3:
                snr_db  = random.uniform(40, 60)
                sig_pow = wav.pow(2).mean().sqrt().clamp(min=1e-8)
                noise   = torch.randn_like(wav) * sig_pow * (10 ** (-snr_db / 20.0))
                wav     = wav + noise

            # Re-clip
            wav = wav.clamp(-1.0, 1.0)

        return wav.squeeze(0)   # [T]


def _collate_fn(batch):
    """Pad batch to max length."""
    lengths = [x.shape[0] for x in batch]
    max_len = max(lengths)
    padded  = torch.zeros(len(batch), max_len)
    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x
    return padded


# ── Lightning Module ──────────────────────────────────────────────────────────

class SeedVCFineTuner(pl.LightningModule):
    """
    Fine-tunes only the reference encoder + diffusion decoder.
    Content encoder is frozen (preserves timbre-leakage resistance).
    """

    def __init__(self, base_ckpt: str, train_dir: str,
                 val_split: float = 0.1, lr: float = 1e-5, batch_size: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.lr         = lr
        self.batch_size = batch_size

        print(f"[Train] Loading Seed-VC checkpoint: {base_ckpt}")
        from seed_vc.model import SeedVC
        self.model = SeedVC.from_pretrained(base_ckpt)

        # Freeze content encoder
        frozen = trainable = 0
        for name, param in self.model.named_parameters():
            if 'content_encoder' in name:
                param.requires_grad = False
                frozen += 1
            else:
                trainable += 1
        print(f"[Train] Frozen={frozen}  Trainable={trainable}")

        # Build dataset
        full_ds  = InstrumentDataset(train_dir, augment=True)
        n_val    = max(1, int(len(full_ds) * val_split))
        n_train  = len(full_ds) - n_val
        self.train_ds, self.val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        # Validation set: no augmentation — re-create without augment
        self.val_ds_clean = InstrumentDataset(train_dir, augment=False)
        print(f"[Train] Train clips={n_train}  Val clips={n_val}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=2, collate_fn=_collate_fn,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=2, collate_fn=_collate_fn,
                          pin_memory=True, persistent_workers=True)

    def training_step(self, batch, batch_idx):
        loss = self.model.compute_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.model.compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr,
                                 betas=(0.9, 0.999), weight_decay=1e-4)
        # Cosine decay with warm-up
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05, anneal_strategy='cos'
        )
        return [opt], [{'scheduler': sched, 'interval': 'step'}]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Seed-VC for a specific instrument",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--instrument', required=True,
                        choices=['piano', 'flute', 'guitar',
                                 'violin', 'cello', 'synth'])
    parser.add_argument('--base_ckpt', default='seed_vc_base.pt')
    parser.add_argument('--epochs',    type=int,   default=200)
    parser.add_argument('--lr',        type=float, default=1e-5)
    parser.add_argument('--batch',     type=int,   default=4)
    parser.add_argument('--precision', default='16-mixed',
                        help='Training precision: 16-mixed | 32 | bf16-mixed')
    args = parser.parse_args()

    train_dir = f"instruments/{args.instrument}/train"
    if not os.path.isdir(train_dir):
        print(f"\n❌ Training directory missing: {train_dir}")
        print(f"   Run:  python prepare_datasets.py --instrument {args.instrument}")
        raise SystemExit(1)

    os.makedirs('checkpoints', exist_ok=True)

    model = SeedVCFineTuner(
        base_ckpt  = args.base_ckpt,
        train_dir  = train_dir,
        lr         = args.lr,
        batch_size = args.batch,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath   = 'checkpoints',
            filename  = f'seedvc_{args.instrument}',
            save_top_k = 1,
            monitor   = 'val_loss',
            mode      = 'min',
        ),
        pl.callbacks.EarlyStopping(
            monitor  = 'val_loss',
            patience = 25,
            mode     = 'min',
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        max_epochs         = args.epochs,
        accelerator        = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices            = 1,
        precision          = args.precision if torch.cuda.is_available() else 32,
        log_every_n_steps  = 5,
        gradient_clip_val  = 1.0,
        callbacks          = callbacks,
        enable_progress_bar = True,
    )

    print(f"\n{'='*56}")
    print("  Seed-VC Instrument Fine-Tuning  v5.3")
    print(f"{'='*56}")
    print(f"  Instrument : {args.instrument}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Batch size : {args.batch}")
    print(f"  Train dir  : {train_dir}")
    print(f"  Checkpoint : checkpoints/seedvc_{args.instrument}.ckpt")
    print()

    trainer.fit(model)

    # Export plain .pt weights
    best_path = f"checkpoints/seedvc_{args.instrument}.ckpt"
    pt_path   = f"checkpoints/seed_vc_{args.instrument}.pt"
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location='cpu')
        torch.save(ckpt['state_dict'], pt_path)
        print(f"\n✅ Fine-tuning complete")
        print(f"   Best val_loss checkpoint: {best_path}")
        print(f"   Plain weights:            {pt_path}")
        print(f"\n   Update config.yaml:")
        print(f"     checkpoints:")
        print(f"       seed_vc:")
        print(f"         {args.instrument}: {pt_path}")


if __name__ == '__main__':
    main()