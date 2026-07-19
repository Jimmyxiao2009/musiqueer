"""
自动踩音 — 批量音乐分析工具
用法:
    python autotag.py 音乐文件.mp3              # 分析单曲
    python autotag.py 音乐文件夹/               # 批量分析目录
    python autotag.py 音乐文件.mp3 --json       # 输出JSON
    python autotag.py 音乐文件.mp3 --plot       # 输出情绪曲线图
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import librosa
import numpy as np
import argparse
from pathlib import Path

import infer
from utils.audio import mel_spectrogram, aggregate_frames


def analyze_file(path, model):
    """分析单个音频文件，返回结构化结果"""
    try:
        r = infer.analyze(model, path)
        if r is None:
            return None
    except Exception as e:
        print(f"  [ERROR] {path}: {e}")
        return None

    name = os.path.splitext(os.path.basename(path))[0]

    # 构建结构化结果
    result = {
        "file": name,
        "path": path,
        "duration_s": round(r["duration"], 1),
        "duration_min": round(r["duration"] / 60, 1),
        "emotion": {
            "Valence": round(r["V_mean"], 3),
            "Arousal": round(r["A_mean"], 3),
            "Tension": round(r["T_mean"], 3),
            "Complexity": round(r["C_mean"], 3),
        },
        "style": r.get("genre_synthesized", ""),
        "moods": [{"tag": m, "prob": round(p, 3)} for m, p in r.get("moods", [])],
        "instruments": [{"name": inst, "prob": round(p, 3)}
                        for inst, p in r.get("instruments", []) if p > 0.1],
    }

    # 时间线摘要（每15秒窗口）
    timeline = []
    for i in range(len(r.get("times", []))):
        tm = r["times"][i]
        seg_m = r["style_curve"][i] if "style_curve" in r else ""
        v = r["V_curve"][i] if "V_curve" in r else 0
        a = r["A_curve"][i] if "A_curve" in r else 0
        t = r["T_curve"][i] if "T_curve" in r else 0
        c = r["C_curve"][i] if "C_curve" in r else 0

        # 检测变化点
        if i > 0:
            prev_m = r["style_curve"][i-1] if "style_curve" in r else ""
            if seg_m != prev_m:
                timeline.append({
                    "time_min": round(tm, 1),
                    "event": f"style change: {prev_m} -> {seg_m}",
                    "emotion": {"V": round(v,2), "A": round(a,2), "T": round(t,2), "C": round(c,2)}
                })

    result["transitions"] = timeline

    return result


def print_result(r, verbose=False):
    """友好打印分析结果"""
    if r is None:
        print("  (无法分析)")
        return

    print(f"\n{'='*55}")
    print(f"  {r['file']}")
    print(f"  时长: {r['duration_min']}min")
    print(f"{'='*55}")

    # 风格
    print(f"\n  >> {r['style']} <<")

    # 情绪
    e = r['emotion']
    print(f"\n  情绪: V={e['Valence']:+.2f}  A={e['Arousal']:+.2f}  T={e['Tension']:+.2f}  C={e['Complexity']:+.2f}")
    print(f"        {'正面' if e['Valence']>0 else '负面'} / {'激烈' if e['Arousal']>0.3 else '平静'} / {'紧张' if e['Tension']>0.2 else '放松'} / {'复杂' if e['Complexity']>0.5 else '简单'}")

    # Mood
    print(f"\n  风格标签:")
    for m in r['moods'][:5]:
        print(f"    {m['tag']:15s} {m['prob']:.0%}")

    # 乐器
    if r['instruments']:
        print(f"\n  乐器:")
        for inst in r['instruments'][:8]:
            print(f"    {inst['name']:20s} {inst['prob']:.0%}")

    # 变化
    if r.get('transitions'):
        print(f"\n  结构变化点:")
        for t in r['transitions'][:10]:
            print(f"    {t['time_min']:.1f}min: {t['event']}")


def plot_emotion_curve(r, output_path):
    """绘制情绪变化曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if "V_curve" not in r or not r["V_curve"]:
            return

        times = [t for t in r.get("times", [])]
        if not times:
            return

        fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        dims = [("V", "Valence", "#2196F3"), ("A", "Arousal", "#F44336"),
                ("T", "Tension", "#FF9800"), ("C", "Complexity", "#9C27B0")]

        for ax, (key, name, color) in zip(axes, dims):
            curve = r.get(f"{key}_curve", [])
            if curve:
                ax.plot(times, curve, color=color, linewidth=1.5)
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                ax.set_ylabel(name, fontsize=10)
                ax.grid(True, alpha=0.2)
                ax.set_ylim(-1.5, 2.5 if key != "C" else 3.5)

        axes[-1].set_xlabel("Time (min)", fontsize=11)
        axes[0].set_title(f"Emotion Curve - {r['file']}  [{r['style']}]", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  曲线图: {output_path}")
        plt.close()
    except Exception as e:
        print(f"  [WARN] 绘图失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="自动踩音 — 音乐自动分析工具")
    parser.add_argument("input", help="音频文件或包含音频文件的目录")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    parser.add_argument("--plot", action="store_true", help="绘制情绪曲线图")
    parser.add_argument("--output", "-o", default=None, help="输出目录（默认与输入同目录）")
    args = parser.parse_args()

    # 收集文件
    input_path = args.input
    if os.path.isdir(input_path):
        files = sorted(sum([glob.glob(os.path.join(input_path, f"*{ext}"))
                          for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg"]], []))
        if not files:
            print(f"目录中没有找到音频文件: {input_path}")
            sys.exit(1)
        print(f"找到 {len(files)} 个音频文件")
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"文件不存在: {input_path}")
        sys.exit(1)

    # 加载模型
    print("加载模型...")
    model = infer.load_model()

    # 输出目录
    out_dir = args.output or os.path.dirname(os.path.abspath(files[0])) if files else "."

    # 分析每个文件
    results = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"\n分析: {name}")

        r = analyze_file(f, model)
        if r is None:
            continue

        results.append(r)

        if args.json:
            json_path = os.path.join(out_dir, f"{name}_analysis.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(r, jf, indent=2, ensure_ascii=False)
            print(f"  JSON: {json_path}")

        if args.plot:
            plot_path = os.path.join(out_dir, f"{name}_curve.png")
            plot_emotion_curve(r, plot_path)

        if not args.json and not args.plot:
            print_result(r)

    # 批量汇总
    if len(results) > 1:
        print(f"\n\n{'='*55}")
        print(f"  批量汇总 ({len(results)} 首)")
        print(f"{'='*55}")
        print(f"{'曲名':20s} {'风格':20s} {'V':>6} {'A':>6} {'T':>6} {'C':>6}  乐器")
        print("-"*85)
        for r in results:
            insts = ", ".join([i['name'] for i in r['instruments'][:3]])
            e = r['emotion']
            print(f"{r['file']:20s} {r['style']:20s} {e['Valence']:>+6.2f} {e['Arousal']:>+6.2f} {e['Tension']:>+6.2f} {e['Complexity']:>+6.2f}  {insts}")

        # 批量JSON
        summary_path = os.path.join(out_dir, "_batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2, ensure_ascii=False)
        print(f"\n批量JSON: {summary_path}")

    print("\n分析完成!")


if __name__ == "__main__":
    main()
