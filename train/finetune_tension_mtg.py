"""
用 MTG-Jamendo mood 标签推导 Tension 代理标签，全量微调 Tension 头
"""

import sys, os, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader

from model.music_understand import MusicUnderstand2
from data.mtg_jamendo import load_metadata, load_tags, CACHE_DIR

CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/tension_finetuned.pth"
OUTPUT_PATH = "checkpoints/tension_mtg_finetuned.pth"
EPOCHS = 15
LR = 5e-5
BATCH_SIZE = 32

# Tension mapping from mood tags (same as before)
TENSION_MAP = {
    "mood/theme---horror": 0.95, "mood/theme---dark": 0.85,
    "mood/theme---heavy": 0.80, "mood/theme---dramatic": 0.75,
    "mood/theme---drama": 0.70, "mood/theme---epic": 0.65,
    "mood/theme---action": 0.70, "mood/theme---powerful": 0.60,
    "mood/theme---sport": 0.55, "mood/theme---trailer": 0.60,
    "mood/theme---game": 0.50, "mood/theme---party": 0.45,
    "mood/theme---energetic": 0.40, "mood/theme---upbeat": 0.30,
    "mood/theme---groovy": 0.25, "mood/theme---fun": 0.25,
    "mood/theme---happy": 0.20, "mood/theme---positive": 0.20,
    "mood/theme---hopeful": 0.15, "mood/theme---inspiring": 0.15,
    "mood/theme---love": 0.10, "mood/theme---romantic": 0.05,
    "mood/theme---melancholic": 0.30, "mood/theme---sad": 0.25,
    "mood/theme---emotional": 0.35, "mood/theme---dream": -0.20,
    "mood/theme---meditative": -0.40, "mood/theme---calm": -0.60,
    "mood/theme---relaxing": -0.50, "mood/theme---soft": -0.50,
    "mood/theme---mellow": -0.40, "mood/theme---slow": -0.30,
    "mood/theme---ambiental": -0.45, "mood/theme---background": -0.40,
    "mood/theme---soundscape": -0.30, "mood/theme---nature": -0.35,
    "mood/theme---space": -0.20, "mood/theme---children": -0.10,
    "mood/theme---christmas": 0.00, "mood/theme---holiday": 0.10,
    "mood/theme---travel": 0.05, "mood/theme---summer": 0.10,
    "mood/theme---sexy": 0.15, "mood/theme---cool": 0.20,
    "mood/theme---retro": 0.00, "mood/theme---advertising": 0.00,
    "mood/theme---commercial": 0.00, "mood/theme---corporate": 0.00,
    "mood/theme---documentary": 0.10, "mood/theme---film": 0.30,
    "mood/theme---movie": 0.30, "mood/theme---adventure": 0.40,
    "mood/theme---ballad": 0.05, "mood/theme---melodic": 0.10,
    "mood/theme---fast": 0.30, "mood/theme---deep": 0.20,
    "mood/theme---motivational": 0.30, "mood/theme---funny": 0.10,
}

# Higher weight for tag combinations that clearly indicate tension level
# If a track has contradictory tags (e.g., both "calm" and "dark"), lower confidence
CONFIDENCE_MAP = {}  # can add per-tag confidence weights


def get_tension_from_tags(tags):
    """Derive tension score from mood tags."""
    matched = [TENSION_MAP.get(t) for t in tags if t in TENSION_MAP]
    if not matched:
        return None, 0.0
    tension = sum(matched) / len(matched)
    # Confidence: higher when more tags agree (low std)
    std = np.std(matched) if len(matched) > 1 else 0.3
    confidence = 1.0 - min(std, 0.5)  # 0.5 to 1.0
    return tension, confidence


class MTGTensionDataset(Dataset):
    """MTG tracks with tension proxy labels from mood tags."""

    def __init__(self, cache_dir: str = CACHE_DIR, max_samples: int = None):
        self.samples = []
        meta = load_metadata()
        
        cached = [int(f[:-3]) for f in os.listdir(cache_dir)
                  if f.endswith(".pt") and f[:-3].isdigit()]
        
        for tid in cached:
            if tid not in meta:
                continue
            tags = meta[tid]["tags"]
            tension, confidence = get_tension_from_tags(tags)
            if tension is None:
                continue
            
            self.samples.append({
                "track_id": tid,
                "tension": tension,
                "confidence": confidence,
            })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"[MTGTension] {len(self.samples)} samples")
        
        # Stats
        tensions = [s["tension"] for s in self.samples]
        confs = [s["confidence"] for s in self.samples]
        print(f"  Tension: mean={np.mean(tensions):+.3f}, std={np.std(tensions):.3f}, "
              f"range=[{np.min(tensions):+.3f}, {np.max(tensions):+.3f}]")
        print(f"  Confidence: mean={np.mean(confs):.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = torch.load(
            os.path.join(CACHE_DIR, f"{s['track_id']}.pt"),
            weights_only=True,
        )
        return {
            "mel": data["mel"],
            "tension": torch.tensor(s["tension"], dtype=torch.float32),
            "confidence": torch.tensor(s["confidence"], dtype=torch.float32),
        }


def main():
    print("Loading config...")
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

    # Freeze all except Tension-related
    for param in model.parameters():
        param.requires_grad = False
    # Output layer: emotion_head.net.8 = Linear(64, 12) -> dim 6,7,8 = T_mean, T_delta, T_var
    for name, param in model.named_parameters():
        if 'emotion_head.net.8' in name:  # Final output layer
            param.requires_grad = True
        if 'emotion_head.net.4' in name:  # Second-to-last layer
            param.requires_grad = True
    
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Dataset
    dataset = MTGTensionDataset(max_samples=5000)  # subset for speed
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )

    print(f"\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            mel = batch["mel"].to(device)
            t_target = batch["tension"].to(device)
            conf = batch["confidence"].to(device)

            out = model(mel, active_experts=["emotion"])
            t_pred = out["emotion"][:, :, 6].mean(dim=1)  # T_mean

            loss = (conf * (t_pred - t_target) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "tension_config": {"source": "MTG_mood_tags", "tension_map": TENSION_MAP},
    }, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Quick test on original 3 tracks
    print("\n=== Quick test ===")
    import librosa
    from utils.audio import mel_spectrogram, aggregate_frames
    model.eval()
    
    tests = [
        ("test_audio/colosseum_bgm.mp3", "Colosseum BGM"),
        ("test_audio/sad_piano.mp3", "Sad Piano"),
        ("test_audio/tense_bgm.mp3", "Tense BGM"),
    ]
    for path, name in tests:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
        yt = torch.from_numpy(y).unsqueeze(0).float()
        mel = aggregate_frames(mel_spectrogram(yt, sr, n_mels=128), 21)
        if mel.shape[-1] > 30: mel = mel[..., :30]
        elif mel.shape[-1] < 30: mel = torch.nn.functional.pad(mel, (0, 30 - mel.shape[-1]))
        with torch.no_grad():
            out = model(mel.unsqueeze(0).to(device), active_experts=["emotion"])
            t = out["emotion"][0, :, 6].mean().item()
        print(f"  {name}: Tension={t:+.3f}")


if __name__ == "__main__":
    main()
