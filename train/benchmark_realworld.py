"""
用真实网络音乐(非DEAM训练/验证集)对情绪模型 + 自动踩谱pipeline做一次跳出训练分布的
sanity benchmark。之前所有评测都在DEAM val集上，这里换成完全没见过的公开授权音乐,
同时测:
  1) 情绪预测方向是否符合人工预期的定性标签(比如"Wallpaper"应该是低Arousal)
  2) 推理速度(real-time factor: 处理耗时 / 音频时长)
  3) 接入chart_generator后的note密度是否随之合理变化

用法:
    python train/benchmark_realworld.py
"""

import os
import sys
import time
import json

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "beatmap"))

from infer_emotion import load_model, load_calibration, analyze_emotion_curve, plot_curve
from chart_generator import generate_chart

AUDIO_DIR = r"C:\Users\OUYANG~1\AppData\Local\Temp\claude\D--Musiclearner\421b3d7a-d579-4e34-a6ab-b41cabb4c43a\scratchpad\benchmark_audio"
OUT_DIR = os.path.join(_root, "docs", "benchmark_realworld")
os.makedirs(OUT_DIR, exist_ok=True)

# 人工定性预期(仅用于sanity-check方向是否合理，不是ground truth标注)
EXPECTED = {
    "Carefree.mp3": {"valence": "positive", "arousal": "medium", "note": "轻快民谣风"},
    "Monkeys Spinning Monkeys.mp3": {"valence": "positive", "arousal": "high", "note": "滑稽/俏皮"},
    "Wallpaper.mp3": {"valence": "neutral", "arousal": "low", "note": "环境背景音乐"},
    "Killers.mp3": {"valence": "negative", "arousal": "high", "note": "黑暗/紧张"},
}


def dir_ok(pred_mean, expected_label, dim):
    if expected_label == "neutral" or expected_label == "medium":
        return None  # 不判定，仅供参考
    if dim == "valence":
        return (pred_mean > 0.05) if expected_label == "positive" else (pred_mean < -0.05)
    if dim == "arousal":
        return (pred_mean > 0.15) if expected_label == "high" else (pred_mean < 0.0)
    return None


def main():
    files = sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith((".mp3", ".wav")))
    if not files:
        print(f"[!] {AUDIO_DIR} 下没有找到音频文件")
        return

    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    model = load_model(os.path.join(_root, "checkpoints", "joint_best_corr_model.pth"))
    valence_calib, arousal_calib = load_calibration(
        os.path.join(_root, "checkpoints", "joint_best_corr_model.calibration.json"))

    rows = []
    for fname in files:
        path = os.path.join(AUDIO_DIR, fname)
        t0 = time.time()
        result = analyze_emotion_curve(path, model, valence_calib, arousal_calib, hop_seconds=5.0)
        emotion_elapsed = time.time() - t0

        v_mean, v_std = float(np.mean(result["valence"])), float(np.std(result["valence"]))
        a_mean, a_std = float(np.mean(result["arousal"])), float(np.std(result["arousal"]))
        rtf = emotion_elapsed / result["duration"]

        plot_path = os.path.join(OUT_DIR, os.path.splitext(fname)[0] + "_curve.png")
        plot_curve(result, plot_path)

        t1 = time.time()
        chart = generate_chart(path, n_lanes=4, difficulty=0.6, emotion_curve=result, emotion_sensitivity=0.4)
        chart_elapsed = time.time() - t1
        note_density = len(chart["notes"]) / chart["duration"]
        lane_counts = np.bincount([n["lane"] for n in chart["notes"]], minlength=4).tolist()

        exp = EXPECTED.get(fname, {})
        v_ok = dir_ok(v_mean, exp.get("valence"), "valence")
        a_ok = dir_ok(a_mean, exp.get("arousal"), "arousal")

        row = {
            "file": fname,
            "duration_s": round(result["duration"], 1),
            "valence_mean": round(v_mean, 3),
            "valence_std": round(v_std, 3),
            "arousal_mean": round(a_mean, 3),
            "arousal_std": round(a_std, 3),
            "emotion_infer_s": round(emotion_elapsed, 2),
            "emotion_rtf": round(rtf, 4),
            "bpm": round(chart["bpm"], 1),
            "note_count": len(chart["notes"]),
            "note_density_per_s": round(note_density, 2),
            "lane_counts": lane_counts,
            "chart_gen_s": round(chart_elapsed, 2),
            "expected": exp,
            "valence_dir_ok": v_ok,
            "arousal_dir_ok": a_ok,
        }
        rows.append(row)

        print(f"\n=== {fname} ===")
        print(f"  时长={row['duration_s']}s  BPM={row['bpm']}")
        print(f"  Valence: mean={row['valence_mean']:+.3f} std={row['valence_std']:.3f}  "
              f"(预期={exp.get('valence','?')}, 方向符合={v_ok})")
        print(f"  Arousal: mean={row['arousal_mean']:+.3f} std={row['arousal_std']:.3f}  "
              f"(预期={exp.get('arousal','?')}, 方向符合={a_ok})")
        print(f"  情绪推理耗时={row['emotion_infer_s']}s  RTF={row['emotion_rtf']}  "
              f"(RTF<1表示比实时快, 数值越小越快)")
        print(f"  自动谱面: {row['note_count']}个note ({row['note_density_per_s']}/s)  "
              f"lane分布={row['lane_counts']}  生成耗时={row['chart_gen_s']}s")

    # 汇总
    n_dir_checks = [r for r in rows if r["valence_dir_ok"] is not None or r["arousal_dir_ok"] is not None]
    n_correct = sum(
        (1 if r["valence_dir_ok"] else 0) + (1 if r["arousal_dir_ok"] else 0)
        for r in rows
    )
    n_total_checks = sum(
        (1 if r["valence_dir_ok"] is not None else 0) + (1 if r["arousal_dir_ok"] is not None else 0)
        for r in rows
    )
    avg_rtf = np.mean([r["emotion_rtf"] for r in rows])

    print("\n" + "=" * 60)
    print(f"汇总: {len(rows)}首曲目, 方向判定 {n_correct}/{n_total_checks} 符合预期定性标签")
    print(f"平均情绪推理RTF={avg_rtf:.4f} (即处理1秒音频平均耗时{avg_rtf*1000:.1f}ms, "
          f"{'快于' if avg_rtf<1 else '慢于'}实时)")

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "avg_rtf": float(avg_rtf),
                   "n_correct": n_correct, "n_total_checks": n_total_checks}, f,
                  ensure_ascii=False, indent=2)
    print(f"[JSON] 完整结果已保存 → {out_json}")


if __name__ == "__main__":
    main()
