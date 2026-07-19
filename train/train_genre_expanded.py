"""
Expanded Genre Head training
Combines: GTZAN (10) + FMA cache + MTG mood-to-genre mapping
"""

import sys, os, torch, yaml, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames

# ── Expanded Genre Categories (20) ──────────────────────────
GENRES = [
    'blues', 'classical', 'country', 'disco', 'electronic',
    'folk', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock', 'instrumental', 'international', 'soul_rnb',
    'experimental', 'spoken', 'ambient', 'soundtrack', 'orchestral',
]
NUM_GENRES = len(GENRES)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

# ── Data sources ────────────────────────────────────────────
GTZAN_DIR = os.path.join(os.path.expanduser("~"),
    ".cache/kagglehub/datasets/andradaolteanu/"
    "gtzan-dataset-music-genre-classification/versions/1/Data/genres_original")
FMA_CACHE = "data/fma_cache"
MTG_CACHE = "data/mtg_cache"

# GTZAN genre → expanded genre mapping
GTZAN_MAP = {
    'blues': 'blues', 'classical': 'classical', 'country': 'country',
    'disco': 'disco', 'hiphop': 'hiphop', 'jazz': 'jazz',
    'metal': 'metal', 'pop': 'pop', 'reggae': 'reggae', 'rock': 'rock',
}

# FMA genre → expanded genre mapping
FMA_MAP = {
    'Rock': 'rock', 'Experimental': 'experimental', 'Electronic': 'electronic',
    'Hip-Hop': 'hiphop', 'Folk': 'folk', 'Pop': 'pop',
    'Instrumental': 'instrumental', 'International': 'international',
    'Classical': 'classical', 'Jazz': 'jazz',
    'Old-Time / Historic': 'folk', 'Spoken': 'spoken',
    'Country': 'country', 'Soul-RnB': 'soul_rnb',
    'Blues': 'blues', 'Easy Listening': 'ambient',
}

# MTG mood tags → genre (weak proxy)
MTG_MOOD_TO_GENRE = {
    'mood/theme---heavy': 'metal',
    'mood/theme---epic': 'orchestral',
    'mood/theme---ambiental': 'ambient',
    'mood/theme---soundscape': 'ambient',
    'mood/theme---space': 'ambient',
    'mood/theme---meditative': 'ambient',
    'mood/theme---relaxing': 'ambient',
    'mood/theme---retro': 'electronic',
    'mood/theme---groovy': 'disco',
    'mood/theme---film': 'soundtrack',
    'mood/theme---movie': 'soundtrack',
    'mood/theme---game': 'soundtrack',
    'mood/theme---drama': 'soundtrack',
    'mood/theme---documentary': 'ambient',
    'mood/theme---corporate': 'ambient',
    'mood/theme---children': 'pop',
    'mood/theme---christmas': 'pop',
    'mood/theme---sad': 'soul_rnb',
    'mood/theme---romantic': 'soul_rnb',
    'mood/theme---love': 'pop',
    'mood/theme---melancholic': 'soul_rnb',
    'mood/theme---fun': 'pop',
    'mood/theme---positive': 'pop',
    'mood/theme---summer': 'pop',
    'mood/theme---hopeful': 'pop',
    'mood/theme---action': 'rock',
    'mood/theme---adventure': 'orchestral',
    'mood/theme---trailer': 'orchestral',
    'mood/theme---sport': 'rock',
    'mood/theme---dramatic': 'orchestral',
    'mood/theme---party': 'electronic',
    'mood/theme---upbeat': 'electronic',
    'mood/theme---energetic': 'electronic',
    'mood/theme---fast': 'metal',
    'mood/theme---powerful': 'rock',
    'mood/theme---cool': 'jazz',
    'mood/theme---ballad': 'soul_rnb',
    'mood/theme---melodic': 'pop',
    'mood/theme---motivational': 'rock',
    'mood/theme---deep': 'electronic',
    'mood/theme---nature': 'ambient',
    'mood/theme---travel': 'folk',
    'mood/theme---sexy': 'soul_rnb',
    'mood/theme---funny': 'pop',
    'mood/theme---holiday': 'pop',
    'mood/theme---inspiring': 'orchestral',
    'mood/theme---uplifting': 'orchestral',
    'mood/theme---background': 'ambient',
    'mood/theme---slow': 'ambient',
    'mood/theme---soft': 'ambient',
    'mood/theme---mellow': 'ambient',
    'mood/theme---dark': 'metal',
    'mood/theme---emotional': 'pop',
    'mood/theme---advertising': 'pop',
    'mood/theme---commercial': 'pop',
}

CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/best_tension_model.pth"
OUTPUT_PATH = "checkpoints/genre_expanded.pth"
EPOCHS = 25
LR = 3e-4

# ── Datasets ────────────────────────────────────────────────
class GenreDataset(Dataset):
    def __init__(self, sources: list):
        """sources: list of (type, path_or_dict)"""
        self.samples = []
        for src_type, src_data in sources:
            if src_type == 'gtzan':
                self._load_gtzan(src_data)
            elif src_type == 'fma_cache':
                self._load_fma_cache(src_data)
            elif src_type == 'mtg_mood':
                self._load_mtg_mood(src_data)
        print(f"[GenreDataset] total: {len(self.samples)} samples, {NUM_GENRES} genres")

    def _load_gtzan(self, genre_dir):
        for gtzan_genre, expanded_genre in GTZAN_MAP.items():
            path = os.path.join(genre_dir, gtzan_genre)
            if not os.path.isdir(path):
                continue
            label = GENRE_TO_IDX[expanded_genre]
            files = glob.glob(os.path.join(path, "*.wav"))
            for f in files:
                self.samples.append((f, label, 'gtzan'))
        print(f"  GTZAN: {len([s for s in self.samples if s[2]=='gtzan'])} samples")

    def _load_fma_cache(self, cache_dir):
        import pandas as pd
        meta_path = os.path.join("data", "fma_meta", "tracks.csv")
        if not os.path.exists(meta_path):
            return
        tracks = pd.read_csv(meta_path, header=[0, 1], low_memory=False)
        tracks.columns = [f"{a}_{b}" if b and "Unnamed" not in str(a) and "Unnamed" not in str(b)
                          else ("track_id" if "Unnamed" in str(a) and "Unnamed" in str(b) else a)
                          for a, b in tracks.columns]
        # Get genre_top column correctly
        genre_col = [c for c in tracks.columns if 'genre_top' in c]
        if not genre_col:
            return
        for fname in os.listdir(cache_dir):
            if not fname.endswith('.pt'):
                continue
            tid = int(fname[:-3])
            row = tracks.loc[tracks['track_id'] == tid]
            if len(row) == 0:
                continue
            genre = row.iloc[0]
            # The genre_top column might be named differently
            for col in genre_col:
                g = row[col].values[0]
                if pd.notna(g) and str(g) in FMA_MAP:
                    label = GENRE_TO_IDX[FMA_MAP[str(g)]]
                    self.samples.append((os.path.join(cache_dir, fname), label, 'fma'))
                    break
        print(f"  FMA cache: {len([s for s in self.samples if s[2]=='fma'])} samples")

    def _load_mtg_mood(self, meta):
        from data.mtg_jamendo import CACHE_DIR
        cache_dir = CACHE_DIR
        for tid, info in meta.items():
            cache_path = os.path.join(cache_dir, f"{tid}.pt")
            if not os.path.exists(cache_path):
                continue
            for tag in info["tags"]:
                if tag in MTG_MOOD_TO_GENRE:
                    genre = MTG_MOOD_TO_GENRE[tag]
                    label = GENRE_TO_IDX[genre]
                    self.samples.append((cache_path, label, 'mtg'))
                    break
        print(f"  MTG mood: {len([s for s in self.samples if s[2]=='mtg'])} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, src = self.samples[idx]
        if src in ('fma', 'mtg'):
            data = torch.load(path, map_location="cpu", weights_only=True)
            mel = data["mel"]
        else:
            try:
                y, sr = librosa.load(path, sr=22050, mono=True, duration=30.0)
            except Exception:
                return self[(idx + 1) % len(self.samples)]
            yt = torch.from_numpy(y).unsqueeze(0).float()
            target_len = int(30.0 * 22050)
            if yt.shape[-1] < target_len:
                yt = torch.nn.functional.pad(yt, (0, target_len - yt.shape[-1]))
            mel = mel_spectrogram(yt, 22050, n_mels=128)
            mel = aggregate_frames(mel, 21)
        # PDebug: print shape mismatch
        if mel.shape[-1] not in (30, 61):
            print(f"  [WARN] {path}: mel.shape={mel.shape}")
        # Align all to 61 frames (MTG standard)
        T = mel.shape[-1]
        if T > 61:
            mel = mel[..., :61]
        elif T < 61:
            mel = torch.nn.functional.pad(mel, (0, 61 - T))

        return mel, torch.tensor(label, dtype=torch.long)


def main():
    print("Loading model...")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build model with expanded genre head
    model = MusicUnderstand2(
        backbone_config=cfg["model"]["backbone"],
        emotion_config={k: v for k, v in cfg["model"]["experts"]["emotion"].items()
                       if k != "loss_weight"},
        mood_config={"num_tags": 59},
        genre_config={"num_genres": NUM_GENRES},
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    # Remove genre_head keys with size mismatch
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if 'genre_head' in k and k in model_state:
            if state_dict[k].shape != model_state[k].shape:
                print(f"  Skip {k}: {state_dict[k].shape} -> {model_state[k].shape}")
                del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.train()

    # Freeze everything except genre head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'genre_head' in name:
            param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Genre head: {NUM_GENRES} genres: {GENRES}")

    # Build dataset
    sources = [('gtzan', GTZAN_DIR)]
    if os.path.isdir(FMA_CACHE) and len(os.listdir(FMA_CACHE)) > 0:
        sources.append(('fma_cache', FMA_CACHE))
    # MTG mood
    try:
        from data.mtg_jamendo import load_metadata
        mtg_meta = load_metadata()
        sources.append(('mtg_mood', mtg_meta))
    except Exception:
        pass

    full_ds = GenreDataset(sources)

    # Train/val split
    n = len(full_ds)
    indices = list(range(n))
    import random
    random.Random(42).shuffle(indices)
    n_val = max(1, int(n * 0.15))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            out = model(mel, active_experts=["genre"])
            loss = torch.nn.functional.cross_entropy(out["genre_logits"], labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (out["genre_logits"].argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for mel, labels in val_loader:
                mel, labels = mel.to(device), labels.to(device)
                logits = model(mel, active_experts=["genre"])["genre_logits"]
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        best_acc = max(best_acc, val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, "
                  f"train_acc={correct/total:.3f}, val_acc={val_acc:.3f}")

    torch.save({"model_state_dict": model.state_dict(), "genre_acc": best_acc,
                "genres": GENRES}, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH} (best val_acc={best_acc:.3f})")

    # Quick test
    print("\n=== Genre Demo ===")
    model.eval()
    tests = [
        ("test_audio/colosseum_bgm.mp3", "Colosseum BGM"),
        ("test_audio/sad_piano.mp3", "Sad Piano"),
        ("test_audio/tense_bgm.mp3", "Tense BGM"),
    ]
    for path, name in tests:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=60)
        segs = []
        t = 0.0
        while t + 15 <= len(y)/sr:
            start = int(t*sr); end = int((t+15)*sr)
            seg = torch.from_numpy(y[start:end]).unsqueeze(0).float()
            mel = aggregate_frames(mel_spectrogram(seg, sr, n_mels=128), 21)
            if mel.shape[-1] > 61: mel = mel[..., :61]
            segs.append(mel.unsqueeze(0))
            t += 7.5
        batch = torch.cat(segs, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch, active_experts=["genre"])["genre_logits"].cpu()
        probs = torch.softmax(logits, dim=1).mean(dim=0)
        top3 = probs.topk(3)
        print(f"\n  {name}:")
        for i in range(3):
            print(f"    {GENRES[top3.indices[i]]}: {top3.values[i]:.1%}")


if __name__ == "__main__":
    main()
