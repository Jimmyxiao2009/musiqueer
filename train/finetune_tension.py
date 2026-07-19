"""
Tension 头微调脚本

用从Bilibili下载的标注数据微调模型 Tension 头。
人工标注的 tension 值范围 [-1, +1]:
  +0.8~+1.0 → 高紧张 (恐怖/悬疑/追逐)
  +0.2~+0.5 → 中等 (战场/史诗)
  -0.5~-0.2 → 低紧张 (平静/悲伤/抒情)
"""

import sys, os, csv, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames

# Config
TRAIN_DIR = "test_audio/train_set"
CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/joint_latest.pth"
OUTPUT_PATH = "checkpoints/tension_finetuned.pth"
EPOCHS = 20
LR = 1e-4
SEGMENT_SEC = 15.0
STRIDE_SEC = 7.5

# Manual labels: filename -> (tension_mean, confidence_weight)
LABELS = {
    # High tension (恐怖/悬疑/追逐) - 清晰的正tension
    "tense_01": (0.9, 1.0),
    "tense_02": (0.8, 1.0),
    "tense_03": (0.9, 1.0),
    "tense_04": (0.7, 1.0),
    # Medium tension (战场/日常)
    "colosseum_bgm": (0.4, 0.7),  # 战场BGM - 气势恢宏但不算恐怖
    "medium_01": (0.2, 0.7),     # 抒情BGM - 略有点情绪波动
    # Low tension (平静/悲伤)
    "calm_01": (-0.3, 1.0),      # 悲伤钢琴
    "calm_02": (-0.4, 1.0),      # 悲伤情绪音乐
}

# Also include the original colosseum from test_audio
import glob
for f in glob.glob("test_audio/colosseum_bgm.mp3"):
    pass  # already in LABELS


class TensionDataset(Dataset):
    """从标注音频中提取片段，给每个片段赋 tension 标签"""

    def __init__(self, train_dir: str, labels: dict):
        self.samples = []

        for fname, (t_label, weight) in labels.items():
            path = os.path.join(train_dir, f"{fname}.mp3")
            if not os.path.exists(path):
                path = os.path.join("test_audio", f"{fname}.mp3")
            if not os.path.exists(path):
                print(f"  [WARN] {fname}.mp3 not found")
                continue

            y, sr = librosa.load(path, sr=22050, mono=True)
            total = len(y) / sr

            # Segment with sliding window
            t = 0.0
            n_seg = 0
            while t + SEGMENT_SEC <= total:
                start = int(t * sr)
                end = int((t + SEGMENT_SEC) * sr)
                seg = torch.from_numpy(y[start:end]).unsqueeze(0).float()

                if seg.shape[-1] < int(SEGMENT_SEC * sr):
                    seg = torch.nn.functional.pad(
                        seg, (0, int(SEGMENT_SEC * sr) - seg.shape[-1]))

                mel = mel_spectrogram(seg, sr, n_mels=128)
                mel = aggregate_frames(mel, 21)
                T = mel.shape[-1]
                if T > 30: mel = mel[..., :30]
                elif T < 30: mel = torch.nn.functional.pad(mel, (0, 30 - T))

                self.samples.append({
                    "mel": mel,
                    "t_label": torch.tensor(t_label, dtype=torch.float32),
                    "weight": weight,
                    "source": fname,
                })
                n_seg += 1
                t += STRIDE_SEC

            print(f"  {fname}: {n_seg} segments, tension={t_label}")

        print(f"  Total: {len(self.samples)} samples from {len(labels)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["mel"], s["t_label"], s["weight"]


def main():
    print("Loading config...")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backbone_cfg = cfg["model"]["backbone"]
    emotion_cfg = {k: v for k, v in cfg["model"]["experts"]["emotion"].items()
                   if k != "loss_weight"}

    model = MusicUnderstand2(backbone_config=backbone_cfg,
                              emotion_config=emotion_cfg,
                              mood_config={"num_tags": 59})
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Freeze all layers except Tension-related parameters
    # In emotion head, dimensions 6,7,8 = T_mean, T_delta, T_var
    # We need to find the output layer weights for these dims
    print("\nModel structure:")
    for name, param in model.named_parameters():
        if 'emotion' in name:
            print(f"  {name}: {param.shape}")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the Tension-related output weights
    # Emotion head output is nn.Sequential with final Linear(64, 12)
    for name, param in model.named_parameters():
        if 'emotion_head' in name and 'net.4' in name:
            # Last linear layer: weight [12, 64], bias [12]
            # Dimensions 6,7,8 correspond to T_mean, T_delta, T_var
            param.requires_grad = True
            print(f"  Unfrozen: {name}")

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset
    print("\nBuilding dataset...")
    dataset = TensionDataset(TRAIN_DIR, LABELS)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"  Batches per epoch: {len(loader)}")

    # Only optimize Tension-related parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )

    # Train
    print(f"\nTraining Tension head (Epochs={EPOCHS}, lr={LR})...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0

        for mel, t_label, weight in loader:
            mel = mel.to(device)
            t_label = t_label.to(device)
            weight = weight.to(device)

            out = model(mel, active_experts=["emotion"])
            emotion = out["emotion"]  # [B, T, 12]

            # Take T_mean (dim 6), average over time
            t_pred = emotion[:, :, 6].mean(dim=1)  # [B]

            # Weighted MSE loss
            loss = (weight * (t_pred - t_label) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Predict on all samples to see progress
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                preds = []
                for mel, _, _ in loader:
                    mel = mel.to(device)
                    out = model(mel, active_experts=["emotion"])
                    p = out["emotion"][:, :, 6].mean(dim=1)
                    preds.append(p.cpu())
                all_preds = torch.cat(preds)
                print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}, "
                      f"pred range=[{all_preds.min():+.3f}, {all_preds.max():+.3f}], "
                      f"pred mean={all_preds.mean():+.3f}")
            model.train()

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "tension_labels": LABELS,
    }, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Quick test on the three original test tracks
    print("\n=== Quick test ===")
    model.eval()
    test_files = [
        ("test_audio/colosseum_bgm.mp3", "斗兽场 BGM"),
        ("test_audio/sad_piano.mp3", "悲伤钢琴"),
        ("test_audio/tense_bgm.mp3", "悬疑紧张"),
    ]
    for path, name in test_files:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=30.0)
        y_t = torch.from_numpy(y).unsqueeze(0).float()
        mel = aggregate_frames(mel_spectrogram(y_t, sr, n_mels=128), 21)
        if mel.shape[-1] > 30: mel = mel[..., :30]
        elif mel.shape[-1] < 30: mel = torch.nn.functional.pad(mel, (0, 30 - mel.shape[-1]))
        with torch.no_grad():
            out = model(mel.unsqueeze(0).to(device), active_experts=["emotion"])
            t = out["emotion"][0, :, 6].mean().item()
        print(f"  {name}: T_mean={t:+.3f}")


if __name__ == "__main__":
    main()
