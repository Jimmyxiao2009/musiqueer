"""
Unified Multi-Task Training
Emotion + Genre(20) + Instrument(9) + Mood(59) + Tension
"""

import sys, os, torch, yaml, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames
from data.deam_dataset import create_dataloaders as create_deam_loaders
from data.mtg_jamendo import load_metadata as load_mtg_meta, CACHE_DIR as MTG_CACHE

# ── Config ───────────────────────────────────────────────────
CONFIG_PATH = "config/default.yaml"
OUTPUT_PATH = "checkpoints/unified_model.pth"
EPOCHS = 20
LR = 3e-4
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Genre categories (20) ────────────────────────────────────
GENRES = [
    'blues','classical','country','disco','electronic',
    'folk','hiphop','jazz','metal','pop',
    'reggae','rock','instrumental','international','soul_rnb',
    'experimental','spoken','ambient','soundtrack','orchestral',
]
NUM_GENRES = len(GENRES)
GENRE_TO_IDX = {g:i for i,g in enumerate(GENRES)}

# GTZAN map
GTZAN_MAP = {
    'blues':'blues','classical':'classical','country':'country',
    'disco':'disco','hiphop':'hiphop','jazz':'jazz',
    'metal':'metal','pop':'pop','reggae':'reggae','rock':'rock',
}

# FMA map
FMA_MAP = {
    'Rock':'rock','Experimental':'experimental','Electronic':'electronic',
    'Hip-Hop':'hiphop','Folk':'folk','Pop':'pop',
    'Instrumental':'instrumental','International':'international',
    'Classical':'classical','Jazz':'jazz',
    'Old-Time / Historic':'folk','Spoken':'spoken',
    'Country':'country','Soul-RnB':'soul_rnb',
    'Blues':'blues','Easy Listening':'ambient',
}

# ── Instrument categories (9) ────────────────────────────────
INSTRUMENTS = ['bass','choir','drums','flute','guitar','piano','strings','synth','vocals']
NUM_INSTR = len(INSTRUMENTS)
INSTR_TO_IDX = {i:idx for idx,i in enumerate(INSTRUMENTS)}

MOOD_TO_INSTR = {
    'mood/theme---heavy':['drums','guitar','bass'],
    'mood/theme---dark':['drums','bass'],
    'mood/theme---epic':['strings','choir','drums'],
    'mood/theme---ambiental':['synth','strings'],
    'mood/theme---soundscape':['synth'],
    'mood/theme---meditative':['strings','piano'],
    'mood/theme---relaxing':['piano','strings'],
    'mood/theme---soft':['piano','strings','vocals'],
    'mood/theme---slow':['piano'],
    'mood/theme---mellow':['piano','guitar'],
    'mood/theme---melodic':['piano','guitar','vocals'],
    'mood/theme---ballad':['piano','vocals','guitar'],
    'mood/theme---romantic':['piano','strings','vocals'],
    'mood/theme---love':['piano','vocals'],
    'mood/theme---sad':['piano','strings','vocals'],
    'mood/theme---emotional':['piano','strings','vocals'],
    'mood/theme---melancholic':['piano','strings'],
    'mood/theme---hopeful':['piano','strings'],
    'mood/theme---inspiring':['strings','choir','piano'],
    'mood/theme---uplifting':['strings','choir','drums'],
    'mood/theme---party':['drums','bass','synth'],
    'mood/theme---fun':['drums','guitar','bass'],
    'mood/theme---groovy':['bass','drums','guitar'],
    'mood/theme---energetic':['drums','bass','guitar'],
    'mood/theme---upbeat':['drums','bass'],
    'mood/theme---fast':['drums','guitar','bass'],
    'mood/theme---sport':['drums','bass'],
    'mood/theme---action':['drums','bass','guitar'],
    'mood/theme---adventure':['strings','drums'],
    'mood/theme---dramatic':['strings','drums','choir'],
    'mood/theme---drama':['strings','piano'],
    'mood/theme---powerful':['drums','bass','choir','strings'],
    'mood/theme---game':['synth','drums'],
    'mood/theme---trailer':['strings','choir','drums','synth'],
    'mood/theme---retro':['synth','drums'],
    'mood/theme---space':['synth','strings'],
    'mood/theme---film':['strings','piano','drums'],
    'mood/theme---movie':['strings','piano'],
    'mood/theme---christmas':['piano','vocals','strings'],
    'mood/theme---children':['piano','vocals'],
    'mood/theme---summer':['guitar','drums','bass'],
    'mood/theme---holiday':['guitar','piano','vocals'],
    'mood/theme---travel':['guitar','drums'],
    'mood/theme---corporate':['piano','guitar'],
    'mood/theme---advertising':['piano','guitar','vocals'],
    'mood/theme---commercial':['piano','guitar'],
    'mood/theme---documentary':['piano','strings'],
    'mood/theme---background':['piano','strings'],
    'mood/theme---calm':['piano','strings'],
    'mood/theme---nature':['piano','strings','flute'],
    'mood/theme---deep':['bass','drums','synth'],
    'mood/theme---cool':['bass','drums','synth'],
    'mood/theme---sexy':['bass','synth','vocals'],
    'mood/theme---positive':['piano','guitar','vocals'],
    'mood/theme---funny':['piano','guitar'],
    'mood/theme---motivational':['drums','bass','choir'],
    'mood/theme---horror':['synth','strings'],
}
FMA_CACHE = "data/fma_cache"
GTZAN_DIR = os.path.join(os.path.expanduser("~"),
    ".cache/kagglehub/datasets/andradaolteanu/"
    "gtzan-dataset-music-genre-classification/versions/1/Data/genres_original")


class UnifiedDataset(Dataset):
    def __init__(self, split="train"):
        self.samples = []  # (mel, genre_label, instr_label, mood_label)
        self.mtg_meta = load_mtg_meta()

        # Load MTG data for mood + instrument
        cached = [int(f[:-3]) for f in os.listdir(MTG_CACHE) if f.endswith(".pt") and f[:-3].isdigit()]
        for i, tid in enumerate(cached):
            if i % 100 < 85 and split != "train": continue
            if i % 100 >= 85 and split != "val": continue
            if tid not in self.mtg_meta: continue
            tags = self.mtg_meta[tid]["tags"]
            instr = torch.zeros(NUM_INSTR)
            for tag in tags:
                if tag in MOOD_TO_INSTR:
                    for inst in MOOD_TO_INSTR[tag]:
                        instr[INSTR_TO_IDX[inst]] = 1.0
            if instr.sum() == 0: continue
            self.samples.append(('mtg', tid, None, instr, tags))

        # Load GTZAN for genre
        for gtzan_g, expanded_g in GTZAN_MAP.items():
            path = os.path.join(GTZAN_DIR, gtzan_g)
            if not os.path.isdir(path): continue
            files = glob.glob(os.path.join(path, "*.wav"))
            for i, f in enumerate(files):
                if i % 100 < 85 and split != "train": continue
                if i % 100 >= 85 and split != "val": continue
                self.samples.append(('gtzan', None, f, None, GENRE_TO_IDX[expanded_g]))

        # Load FMA for genre
        if os.path.isdir(FMA_CACHE):
            import pandas as pd
            meta_path = "data/fma_meta/tracks.csv"
            if os.path.exists(meta_path):
                fma_meta = pd.read_csv(meta_path, header=[0,1], low_memory=False)
                fma_meta.columns = ['track_id' if 'Unnamed' in str(a) and 'Unnamed' in str(b)
                    else f"{a}_{b}" if b and 'Unnamed' not in str(b) else a
                    for a,b in fma_meta.columns]
                for fname in os.listdir(FMA_CACHE):
                    if not fname.endswith('.pt'): continue
                    tid = int(fname[:-3])
                    row = fma_meta[fma_meta['track_id'] == tid]
                    if len(row) == 0: continue
                    g = row.iloc[0].get('track_genre_top', None)
                    if pd.notna(g) and str(g) in FMA_MAP:
                        self.samples.append(('fma', None, os.path.join(FMA_CACHE, fname), None, GENRE_TO_IDX[FMA_MAP[str(g)]]))

        print(f"[Unified {split}] {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        src, tid, path, extra, label = self.samples[idx]
        
        if src == 'mtg':
            data = torch.load(os.path.join(MTG_CACHE, f"{tid}.pt"), map_location="cpu", weights_only=True)
            mel = data["mel"]
            instr = extra
            genre = torch.zeros(NUM_GENRES)  # no genre label
            # mood label from tags
            mood = torch.zeros(59)
            for tag in label:
                from data.mtg_jamendo import load_tags
                tag_list = load_tags()
                if tag in tag_list:
                    mood[tag_list.index(tag)] = 1.0
            
        elif src == 'gtzan':
            try:
                import librosa
                y, sr = librosa.load(path, sr=22050, mono=True, duration=30.0)
            except:
                return self[(idx+1) % len(self)]
            yt = torch.from_numpy(y).unsqueeze(0).float()
            tl = 30 * 22050
            if yt.shape[-1] < tl: yt = torch.nn.functional.pad(yt, (0, tl - yt.shape[-1]))
            mel = mel_spectrogram(yt, 22050, n_mels=128)
            mel = aggregate_frames(mel, 21)
            instr = torch.zeros(NUM_INSTR)
            mood = torch.zeros(59)
            genre = torch.zeros(NUM_GENRES)
            genre[label] = 1.0
            
        elif src == 'fma':
            data = torch.load(path, map_location="cpu", weights_only=True)
            mel = data["mel"]
            instr = torch.zeros(NUM_INSTR)
            mood = torch.zeros(59)
            genre = torch.zeros(NUM_GENRES)
            genre[label] = 1.0

        # Align mel frames
        T = mel.shape[-1]
        if T > 61: mel = mel[..., :61]
        elif T < 61: mel = torch.nn.functional.pad(mel, (0, 61 - T))

        return mel, genre, instr, mood


def main():
    print(f"Device: {DEVICE}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MusicUnderstand2(
        backbone_config=cfg["model"]["backbone"],
        emotion_config={k:v for k,v in cfg["model"]["experts"]["emotion"].items() if k != "loss_weight"},
        mood_config={"num_tags": 59},
        genre_config={"num_genres": NUM_GENRES},
        instrument_config={"num_instruments": NUM_INSTR},
    )

    # Load best available checkpoint
    for ckpt_path in ['checkpoints/joint_latest.pth', 'checkpoints/best_tension_model.pth']:
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"]
            # Remove mismatched keys
            for k in list(sd.keys()):
                if k in model.state_dict() and sd[k].shape != model.state_dict()[k].shape:
                    del sd[k]
            model.load_state_dict(sd, strict=False)
            print(f"Loaded: {ckpt_path}")
            break

    model.to(DEVICE)
    model.train()

    train_ds = UnifiedDataset("train")
    val_ds = UnifiedDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for mel, genre, instr, mood in train_loader:
            mel = mel.to(DEVICE)
            genre, instr, mood = genre.to(DEVICE), instr.to(DEVICE), mood.to(DEVICE)

            out = model(mel, active_experts=["emotion", "genre", "mood", "instrument"])

            loss = 0
            # Genre loss
            if genre.sum() > 0:
                loss += torch.nn.functional.cross_entropy(out["genre_logits"], genre.argmax(dim=1))
            # Instrument loss
            loss += torch.nn.functional.binary_cross_entropy_with_logits(out["instrument_logits"], instr)
            # Mood loss
            loss += torch.nn.functional.binary_cross_entropy_with_logits(out["mood_logits"], mood)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mel, genre, instr, mood in val_loader:
                mel = mel.to(DEVICE)
                genre, instr, mood = genre.to(DEVICE), instr.to(DEVICE), mood.to(DEVICE)
                out = model(mel, active_experts=["emotion", "genre", "mood", "instrument"])
                l = torch.nn.functional.binary_cross_entropy_with_logits(out["instrument_logits"], instr)
                l += torch.nn.functional.binary_cross_entropy_with_logits(out["mood_logits"], mood)
                if genre.sum() > 0:
                    l += torch.nn.functional.cross_entropy(out["genre_logits"], genre.argmax(dim=1))
                val_loss += l.item()

        val_loss /= len(val_loader)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss}, OUTPUT_PATH)
            print(f"  * Saved (val_loss={val_loss:.4f})")

    print(f"\nDone! Best val_loss={best_loss:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
