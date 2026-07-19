"""
Instrument Head 训练 — MTG mood→instrument 映射
"""

import sys, os, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from model.music_understand import MusicUnderstand2
from data.mtg_jamendo import load_metadata, CACHE_DIR

# Instrument config
MOOD_TO_INSTR = {
    'mood/theme---heavy': ['drums','guitar','bass'],
    'mood/theme---dark': ['drums','bass'],
    'mood/theme---epic': ['strings','choir','drums'],
    'mood/theme---ambiental': ['synth','strings'],
    'mood/theme---soundscape': ['synth'],
    'mood/theme---meditative': ['strings','piano'],
    'mood/theme---relaxing': ['piano','strings'],
    'mood/theme---soft': ['piano','strings','vocals'],
    'mood/theme---slow': ['piano'],
    'mood/theme---mellow': ['piano','guitar'],
    'mood/theme---melodic': ['piano','guitar','vocals'],
    'mood/theme---ballad': ['piano','vocals','guitar'],
    'mood/theme---romantic': ['piano','strings','vocals'],
    'mood/theme---love': ['piano','vocals'],
    'mood/theme---sad': ['piano','strings','vocals'],
    'mood/theme---emotional': ['piano','strings','vocals'],
    'mood/theme---melancholic': ['piano','strings'],
    'mood/theme---hopeful': ['piano','strings'],
    'mood/theme---inspiring': ['strings','choir','piano'],
    'mood/theme---uplifting': ['strings','choir','drums'],
    'mood/theme---party': ['drums','bass','synth'],
    'mood/theme---fun': ['drums','guitar','bass'],
    'mood/theme---groovy': ['bass','drums','guitar'],
    'mood/theme---energetic': ['drums','bass','guitar'],
    'mood/theme---upbeat': ['drums','bass'],
    'mood/theme---fast': ['drums','guitar','bass'],
    'mood/theme---sport': ['drums','bass'],
    'mood/theme---action': ['drums','bass','guitar'],
    'mood/theme---adventure': ['strings','drums'],
    'mood/theme---dramatic': ['strings','drums','choir'],
    'mood/theme---drama': ['strings','piano'],
    'mood/theme---powerful': ['drums','bass','choir','strings'],
    'mood/theme---game': ['synth','drums'],
    'mood/theme---trailer': ['strings','choir','drums','synth'],
    'mood/theme---retro': ['synth','drums'],
    'mood/theme---space': ['synth','strings'],
    'mood/theme---film': ['strings','piano','drums'],
    'mood/theme---movie': ['strings','piano'],
    'mood/theme---christmas': ['piano','vocals','strings'],
    'mood/theme---children': ['piano','vocals'],
    'mood/theme---summer': ['guitar','drums','bass'],
    'mood/theme---holiday': ['guitar','piano','vocals'],
    'mood/theme---travel': ['guitar','drums'],
    'mood/theme---corporate': ['piano','guitar'],
    'mood/theme---advertising': ['piano','guitar','vocals'],
    'mood/theme---commercial': ['piano','guitar'],
    'mood/theme---documentary': ['piano','strings'],
    'mood/theme---background': ['piano','strings'],
    'mood/theme---calm': ['piano','strings'],
    'mood/theme---nature': ['piano','strings','flute'],
    'mood/theme---deep': ['bass','drums','synth'],
    'mood/theme---cool': ['bass','drums','synth'],
    'mood/theme---sexy': ['bass','synth','vocals'],
    'mood/theme---positive': ['piano','guitar','vocals'],
    'mood/theme---funny': ['piano','guitar'],
    'mood/theme---motivational': ['drums','bass','choir'],
    'mood/theme---horror': ['synth','strings'],
}

ALL_INSTRUMENTS = sorted(set(sum(MOOD_TO_INSTR.values(), [])))
NUM_INSTR = len(ALL_INSTRUMENTS)
INSTR_TO_IDX = {i:idx for idx,i in enumerate(ALL_INSTRUMENTS)}

CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/genre_expanded.pth"
OUTPUT_PATH = "checkpoints/instrument_mtg.pth"
EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 32
MAX_SAMPLES = 10000


class MTGInstrumentDataset(Dataset):
    def __init__(self, split="train"):
        self.samples = []
        meta = load_metadata()
        cached = [int(f[:-3]) for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and f[:-3].isdigit()]
        cached = sorted(cached)

        for i, tid in enumerate(cached):
            if i >= MAX_SAMPLES:
                break
            if tid not in meta:
                continue
            tags = meta[tid]["tags"]
            label = torch.zeros(NUM_INSTR)
            has_label = False
            for tag in tags:
                if tag in MOOD_TO_INSTR:
                    for inst in MOOD_TO_INSTR[tag]:
                        label[INSTR_TO_IDX[inst]] = 1.0
                        has_label = True
            if not has_label:
                continue

            # 85/15 split
            if i % 100 < 85:
                if split != "train": continue
            else:
                if split != "val": continue

            self.samples.append(tid)

        print(f"[MTGInstr {split}] {len(self.samples)} samples, {NUM_INSTR} instruments")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tid = self.samples[idx]
        data = torch.load(os.path.join(CACHE_DIR, f"{tid}.pt"), map_location="cpu", weights_only=True)
        mel = data["mel"]

        # Get label
        meta = load_metadata()
        label = torch.zeros(NUM_INSTR)
        for tag in meta[tid]["tags"]:
            if tag in MOOD_TO_INSTR:
                for inst in MOOD_TO_INSTR[tag]:
                    label[INSTR_TO_IDX[inst]] = 1.0

        T = mel.shape[-1]
        if T > 61: mel = mel[..., :61]
        elif T < 61: mel = torch.nn.functional.pad(mel, (0, 61 - T))

        return mel, label


class FastMTGDataset(Dataset):
    """Pre-load labels for speed"""
    def __init__(self, split="train"):
        from functools import cache
        self.data = []
        meta = load_metadata()
        cached = sorted([int(f[:-3]) for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and f[:-3].isdigit()])

        for i, tid in enumerate(cached):
            if i >= MAX_SAMPLES: break
            if tid not in meta: continue
            label = torch.zeros(NUM_INSTR)
            has = False
            for tag in meta[tid]["tags"]:
                if tag in MOOD_TO_INSTR:
                    for inst in MOOD_TO_INSTR[tag]:
                        label[INSTR_TO_IDX[inst]] = 1.0
                        has = True
            if not has: continue
            if i % 100 < 85 and split != "train": continue
            if i % 100 >= 85 and split != "val": continue
            self.data.append((tid, label))

        print(f"[FastMTG {split}] {len(self.data)} samples")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        tid, label = self.data[idx]
        data = torch.load(os.path.join(CACHE_DIR, f"{tid}.pt"), map_location="cpu", weights_only=True)
        mel = data["mel"]
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
        emotion_config={k: v for k, v in cfg["model"]["experts"]["emotion"].items() if k != "loss_weight"},
        mood_config={"num_tags": 59},
        instrument_config={"num_instruments": NUM_INSTR},
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]
    for k in list(sd.keys()):
        if 'instrument_head' in k or 'genre_head' in k:
            if k in model.state_dict() and sd[k].shape != model.state_dict()[k].shape:
                del sd[k]
    model.load_state_dict(sd, strict=False)

    for p in model.parameters(): p.requires_grad = False
    for n, p in model.named_parameters():
        if 'instrument_head' in n: p.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Instruments: {ALL_INSTRUMENTS}")

    train_ds = FastMTGDataset("train")
    val_ds = FastMTGDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-5)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            logits = model(mel, active_experts=["instrument"])["instrument_logits"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds, tgts = [], []
            for mel, labels in val_loader:
                logits = model(mel.to(device), active_experts=["instrument"])["instrument_logits"]
                probs = torch.sigmoid(logits).cpu()
                preds.append((probs > 0.5).float())
                tgts.append(labels)

        if preds:
            ap = torch.cat(preds); at = torch.cat(tgts)
            tp = (ap*at).sum(0); fp = (ap*(1-at)).sum(0); fn = ((1-ap)*at).sum(0)
            p_ = tp/(tp+fp+1e-8); r_ = tp/(tp+fn+1e-8); f1 = (2*p_*r_/(p_+r_+1e-8))
            mf1 = f1.mean().item()
            if mf1 > best_f1: best_f1 = mf1
            if epoch % 3 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}, val_F1={mf1:.3f}")
                for i, inst in enumerate(ALL_INSTRUMENTS):
                    print(f"    {inst:8s}: P={p_[i]:.2f} R={r_[i]:.2f} F1={f1[i]:.2f}")

    torch.save({"model_state_dict": model.state_dict(), "instruments": ALL_INSTRUMENTS, "best_f1": best_f1}, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH} (best val_F1={best_f1:.3f})")

if __name__ == "__main__":
    main()
