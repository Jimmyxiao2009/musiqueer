"""
MusicCaps 乐器检测训练 — 训练 Extended InstrumentHead
"""

import sys, os, torch, yaml, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames

# ── Expanded Instrument Categories (16) ──────────────────────
INSTRUMENTS = [
    'piano', 'drums', 'bass', 'guitar', 'violin', 'cello',
    'flute', 'trumpet', 'saxophone', 'vocals', 'choir',
    'strings', 'synth', 'organ', 'harp', 'snare',
]
NUM_INSTRUMENTS = len(INSTRUMENTS)
INSTR_TO_IDX = {i: idx for idx, i in enumerate(INSTRUMENTS)}

# Mapping from keyword to instrument label
INSTR_KEYWORDS = {
    'piano': ['piano', 'keyboard', 'keys'],
    'drums': ['drums', 'drum', 'drumming', 'percussion', 'kick', 'hi-hat', 'cymbal'],
    'bass': ['bass'],
    'guitar': ['guitar'],
    'violin': ['violin', 'fiddle'],
    'cello': ['cello'],
    'flute': ['flute', 'pipe'],
    'trumpet': ['trumpet', 'horn', 'brass'],
    'saxophone': ['saxophone', 'sax'],
    'vocals': ['vocal', 'singing', 'voice', 'vocalist', 'singer'],
    'choir': ['choir', 'choral', 'chorus'],
    'strings': ['strings', 'string ensemble', 'orchestra', 'orchestral'],
    'synth': ['synth', 'synthesizer', 'electronic'],
    'organ': ['organ'],
    'harp': ['harp'],
    'snare': ['snare', 'snare drum'],
}

CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/genre_expanded.pth"
OUTPUT_PATH = "checkpoints/instrument_trained.pth"
CACHE_DIR = "data/musiccaps_cache"
EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 32
MAX_SAMPLES = 5000

class MusicCapsInstrumentDataset(Dataset):
    """MusicCaps-derived instrument labels (multi-label)"""

    def __init__(self, split="train", max_samples=MAX_SAMPLES):
        self.samples = []
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load from HF
        ds = load_dataset('google/MusicCaps', split='train', streaming=True)

        for i, sample in enumerate(ds):
            if i >= max_samples:
                break

            ytid = sample['ytid']
            cache_path = os.path.join(self.cache_dir, f"{ytid}.pt")

            # Get multi-label from aspect_list + caption
            text = ' '.join(sample.get('aspect_list', [])) + ' ' + sample.get('caption', '')
            text = text.lower()

            label = torch.zeros(NUM_INSTRUMENTS)
            has_label = False
            for inst, keywords in INSTR_KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        label[INSTR_TO_IDX[inst]] = 1.0
                        has_label = True
                        break

            if not has_label:
                continue

            # Determine split (use last 15% as val)
            if i % 100 < 85:
                if split != "train":
                    continue
            else:
                if split != "val":
                    continue

            # Check if we have cached mel
            if os.path.exists(cache_path):
                self.samples.append((cache_path, label, True))
            else:
                # Store YouTube ID for later download
                self.samples.append((ytid, label, False))

        print(f"[MusicCaps {split}] {len(self.samples)} samples, {NUM_INSTRUMENTS} instruments")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_or_id, label, is_cached = self.samples[idx]

        if is_cached:
            data = torch.load(path_or_id, map_location="cpu", weights_only=True)
            mel = data["mel"]
        else:
            # Stream from YouTube
            import subprocess, io, librosa
            ytid = path_or_id
            url = f"https://www.youtube.com/watch?v={ytid}"
            try:
                cmd = ['yt-dlp', '-x', '--audio-format', 'wav',
                       '--output', '-', '--download-sections', '*0-30',
                       '--force-keyframes-at-cuts', '--no-playlist',
                       '--quiet', '--no-warnings', url]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                audio, sr = librosa.load(io.BytesIO(proc.stdout.read()),
                                          sr=22050, mono=True)
                proc.stdout.close()
                proc.wait(timeout=30)
                yt = torch.from_numpy(audio).unsqueeze(0).float()
                target_len = 30 * 22050
                if yt.shape[-1] > target_len: yt = yt[..., :target_len]
                elif yt.shape[-1] < target_len: yt = torch.nn.functional.pad(yt, (0, target_len - yt.shape[-1]))
                mel = mel_spectrogram(yt, 22050, n_mels=128)
                mel = aggregate_frames(mel, 21)
                # Cache for future
                torch.save({"mel": mel}, os.path.join(self.cache_dir, f"{ytid}.pt"))
            except Exception as e:
                return self[(idx + 1) % len(self.samples)]

        T = mel.shape[-1]
        if T > 61: mel = mel[..., :61]
        elif T < 61: mel = torch.nn.functional.pad(mel, (0, 61 - T))

        return mel, label


def main():
    print("Loading model...")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MusicUnderstand2(
        backbone_config=cfg["model"]["backbone"],
        emotion_config={k: v for k, v in cfg["model"]["experts"]["emotion"].items()
                       if k != "loss_weight"},
        mood_config={"num_tags": 59},
        instrument_config={"num_instruments": NUM_INSTRUMENTS},
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    # Remove instrument_head keys (new size)
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if 'instrument_head' in k or 'genre_head' in k:
            if k in model_state and state_dict[k].shape != model_state[k].shape:
                print(f"  Skip {k}: {state_dict[k].shape} -> {model_state[k].shape}")
                del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # Freeze all except instrument head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'instrument_head' in name:
            param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nInstruments: {INSTRUMENTS}")

    # Data
    train_ds = MusicCapsInstrumentDataset(split="train")
    val_ds = MusicCapsInstrumentDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )

    # Train
    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            out = model(mel, active_experts=["instrument"])
            logits = out["instrument_logits"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Val
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                out = model(mel, active_experts=["instrument"])
                probs = torch.sigmoid(out["instrument_logits"]).cpu()
                preds.append((probs > 0.5).float())
                targets.append(labels)

        if preds:
            all_preds = torch.cat(preds)
            all_targets = torch.cat(targets)
            tp = (all_preds * all_targets).sum(dim=0)
            fp = (all_preds * (1 - all_targets)).sum(dim=0)
            fn = ((1 - all_preds) * all_targets).sum(dim=0)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            mean_f1 = f1.mean().item()

            if mean_f1 > best_f1:
                best_f1 = mean_f1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, val_F1={mean_f1:.3f}")
                for i, inst in enumerate(INSTRUMENTS):
                    if f1[i] > 0.1:
                        print(f"      {inst:10s}: P={precision[i]:.2f} R={recall[i]:.2f} F1={f1[i]:.2f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "instruments": INSTRUMENTS,
        "best_f1": best_f1,
    }, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH} (best val_F1={best_f1:.3f})")


if __name__ == "__main__":
    main()
