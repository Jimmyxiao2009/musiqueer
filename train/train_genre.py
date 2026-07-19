"""
GTZAN 体裁分类训练 — 训练 GenreHead + 推理
"""

import sys, os, torch, yaml, glob, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames
from model.experts.genre_head import GENRES, NUM_GENRES

# GTZAN data path
GTZAN_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets", "andradaolteanu",
    "gtzan-dataset-music-genre-classification", "versions", "1", "Data"
)
GENRE_DIR = os.path.join(GTZAN_PATH, "genres_original")

CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/best_tension_model.pth"
OUTPUT_PATH = "checkpoints/genre_trained.pth"
EPOCHS = 20
LR = 3e-4
BATCH_SIZE = 16


class GTZANDataset(Dataset):
    """GTZAN genre classification dataset with mel extraction"""

    def __init__(self, genre_dir: str, split: str = "train", val_ratio: float = 0.2,
                 max_duration: float = 30.0, seed: int = 42):
        self.max_duration = max_duration
        self.samples = []

        for i, genre in enumerate(GENRES):
            genre_path = os.path.join(genre_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            files = sorted(glob.glob(os.path.join(genre_path, "*.wav")))
            random.Random(seed).shuffle(files)
            n_val = max(1, int(len(files) * val_ratio))

            if split == "train":
                files = files[n_val:]
            else:
                files = files[:n_val]

            for f in files:
                self.samples.append((f, i))

        print(f"[GTZAN {split}] {len(self.samples)} samples, "
              f"genres={len(GENRES)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            y, sr = librosa.load(path, sr=22050, mono=True, duration=self.max_duration)
        except Exception:
            # If file corrupt, return a nearby valid sample
            return self[(idx + 1) % len(self.samples)]
        yt = torch.from_numpy(y).unsqueeze(0).float()

        target_len = int(self.max_duration * sr)
        if yt.shape[-1] < target_len:
            yt = torch.nn.functional.pad(yt, (0, target_len - yt.shape[-1]))

        mel = mel_spectrogram(yt, sr, n_mels=128)
        mel = aggregate_frames(mel, 21)
        T = mel.shape[-1]
        if T > 30: mel = mel[..., :30]
        elif T < 30: mel = torch.nn.functional.pad(mel, (0, 30 - T))

        return mel, torch.tensor(label, dtype=torch.long)


def main():
    print("Loading model...")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MusicUnderstand2(
        backbone_config=cfg["model"]["backbone"],
        emotion_config={k: v for k, v in cfg["model"]["experts"]["emotion"].items()
                       if k != "loss_weight"},
        mood_config={"num_tags": 59},
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Freeze all except genre head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'genre_head' in name:
            param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nTrainable params:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # Data
    train_ds = GTZANDataset(GENRE_DIR, split="train")
    val_ds = GTZANDataset(GENRE_DIR, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )

    # Train
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            out = model(mel, active_experts=["genre"])
            logits = out["genre_logits"]

            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for mel, labels in val_loader:
                mel, labels = mel.to(device), labels.to(device)
                out = model(mel, active_experts=["genre"])
                logits = out["genre_logits"]
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, "
                  f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "genre_acc": best_acc,
    }, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH} (best val_acc={best_acc:.3f})")

    # ============================================================
    # Combined inference on test tracks
    # ============================================================
    print("\n=== Genre + Emotion + Tension Demo ===")
    model.eval()

    test_tracks = [
        ("test_audio/colosseum_bgm.mp3", "Colosseum BGM"),
        ("test_audio/sad_piano.mp3", "Sad Piano"),
        ("test_audio/tense_bgm.mp3", "Tense BGM"),
    ]

    for path, name in test_tracks:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=60)

        # Sliding window for time-resolved analysis
        segs, times = [], []
        t = 0.0
        while t + 15 <= len(y) / sr:
            start = int(t * sr)
            end = int((t + 15) * sr)
            seg = torch.from_numpy(y[start:end]).unsqueeze(0).float()
            mel = aggregate_frames(mel_spectrogram(seg, sr, n_mels=128), 21)
            if mel.shape[-1] > 30: mel = mel[..., :30]
            segs.append(mel.unsqueeze(0))
            times.append(t)
            t += 7.5

        batch = torch.cat(segs, dim=0).to(device)
        with torch.no_grad():
            out = model(batch, active_experts=["emotion", "genre"])
            emotion = out["emotion"].cpu()
            genre_logits = out["genre_logits"].cpu()

        genre_probs = torch.softmax(genre_logits, dim=1)
        top_genre = genre_probs.mean(dim=0).topk(3)

        v = emotion[:, :, 0].mean().item()
        a = emotion[:, :, 3].mean().item()
        t_val = emotion[:, :, 6].mean().item()
        c = emotion[:, :, 9].mean().item()

        print(f"\n  {name}:")
        print(f"    Emotion: V={v:+.2f} A={a:+.2f} T={t_val:+.2f} C={c:+.2f}")
        print(f"    Top genres:")
        for i in range(3):
            g_idx = top_genre.indices[i].item()
            g_prob = top_genre.values[i].item()
            print(f"      {GENRES[g_idx]}: {g_prob:.1%}")


if __name__ == "__main__":
    main()
