"""
MusicUnderstand — Unified Inference Demo
用法: python infer.py <audio_path>
"""

import sys, os, torch, yaml, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import librosa
import numpy as np

from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames
from model.experts.genre_head import GENRES as GENRES_10
from data.mtg_jamendo import load_tags

# ── Config ───────────────────────────────────────────────────
CONFIG_PATH = "config/default.yaml"
CKPT_PATH = "checkpoints/unified.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expanded 20 genres
GENRES_20 = [
    'blues', 'classical', 'country', 'disco', 'electronic',
    'folk', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock', 'instrumental', 'international', 'soul_rnb',
    'experimental', 'spoken', 'ambient', 'soundtrack', 'orchestral',
]

# 20 instruments from OpenMIC
INSTRUMENTS = ['accordion','banjo','bass','cello','clarinet','cymbals','drums','flute',
    'guitar','mallet_percussion','mandolin','organ','piano','saxophone','synthesizer',
    'trombone','trumpet','ukulele','violin','voice']

# MTG mood tags (59)
MOOD_TAGS = load_tags()


def load_model():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MusicUnderstand2(
        backbone_config=cfg["model"]["backbone"],
        emotion_config={k: v for k, v in cfg["model"]["experts"]["emotion"].items() if k != "loss_weight"},
        mood_config={"num_tags": 59},
        genre_config={"num_genres": 20},
        instrument_config={"num_instruments": 20},
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]

    for k in list(sd.keys()):
        if k in model.state_dict() and sd[k].shape != model.state_dict()[k].shape:
            del sd[k]
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def analyze(model, audio_path, segment_sec=15.0, stride_sec=7.5):
    """Full track analysis with sliding window."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    total_sec = len(y) / sr

    segments, times = [], []
    t = 0.0
    while t + segment_sec <= total_sec:
        start = int(t * sr)
        end = int((t + segment_sec) * sr)
        seg = torch.from_numpy(y[start:end]).unsqueeze(0).float()

        if seg.shape[-1] < int(segment_sec * sr):
            seg = torch.nn.functional.pad(seg, (0, int(segment_sec * sr) - seg.shape[-1]))

        mel = mel_spectrogram(seg, sr, n_mels=128)
        mel = aggregate_frames(mel, 21)
        T = mel.shape[-1]
        if T > 30: mel = mel[..., :30]
        elif T < 30: mel = torch.nn.functional.pad(mel, (0, 30 - T))

        segments.append(mel.unsqueeze(0))
        times.append(t)
        t += stride_sec

    if not segments:
        return None

    batch = torch.cat(segments, dim=0).to(DEVICE)

    with torch.no_grad():
        out = model(batch, active_experts=["emotion", "genre", "mood", "instrument"])

    results = {}
    times_min = [tm / 60 for tm in times]

    # Emotion (V/A/T/C)
    emo = out["emotion"].cpu()
    for idx, name in [(0, "V"), (3, "A"), (6, "T"), (9, "C")]:
        vals = emo[:, :, idx].mean(dim=1).numpy()
        results[f"{name}_mean"] = vals.mean().item()
        results[f"{name}_curve"] = vals.tolist()

    mood_names = [MOOD_TAGS[i].replace("mood/theme---", "") for i in range(len(MOOD_TAGS))]

    # ============================================================
    # MOOD-based genre/style synthesis
    # Combines mood tags + emotion into descriptive genre labels
    # ============================================================
    mood_logits = out["mood_logits"].cpu()
    mood_probs = torch.sigmoid(mood_logits)
    avg_mood = mood_probs.mean(dim=0)
    mood_dict = {mood_names[i]: avg_mood[i].item() for i in range(len(MOOD_TAGS))}
    
    top5_mood = avg_mood.topk(5)
    results["moods"] = [(mood_names[i], top5_mood.values[i].item())
                        for i in range(5)]

    # Synthesize genre from mood + emotion
    a = results["A_mean"]
    t = results["T_mean"]
    
    has = lambda m: mood_dict.get(m, 0) > 0.25
    c_val = results["C_mean"]
    
    # Check instrument probabilities for genre hints
    avg_instr_check = torch.sigmoid(out["instrument_logits"].cpu()).mean(dim=0)
    instr_idx = lambda name: INSTRUMENTS.index(name)
    violin_p = avg_instr_check[instr_idx("violin")].item()
    cello_p = avg_instr_check[instr_idx("cello")].item()
    piano_p = avg_instr_check[instr_idx("piano")].item()
    voice_p = avg_instr_check[instr_idx("voice")].item()
    drums_p = avg_instr_check[instr_idx("drums")].item()
    cymbals_p = avg_instr_check[instr_idx("cymbals")].item()
    bass_p = avg_instr_check[instr_idx("bass")].item()
    sax_p = avg_instr_check[instr_idx("saxophone")].item()
    trumpet_p = avg_instr_check[instr_idx("trumpet")].item()
    guitar_p = avg_instr_check[instr_idx("guitar")].item()
    synth_p = avg_instr_check[instr_idx("synthesizer")].item()
    
    # Determine primary genre
    # Classical first (instrument-based)
    if (violin_p > 0.2 or cello_p > 0.15) and c_val > 0.5 and 0.2 < a < 0.6:
        synth = "Classical/Baroque"
    elif piano_p > 0.3 and a < 0.2:
        synth = "Classical/Piano"
    elif voice_p > 0.3 and has("epic"):
        synth = "Choral/Epic"
    # Jazz: saxophone or trumpet, moderate energy
    elif (sax_p > 0.15 or trumpet_p > 0.15) and 0.1 < a < 0.6:
        if drums_p > 0.3:
            synth = "Jazz/Big Band"
        else:
            synth = "Jazz/Swing"
    # Rock: drums + voice/guitar + high arousal + high tension
    elif (drums_p > 0.15 or cymbals_p > 0.15) and (voice_p > 0.15 or guitar_p > 0.1) and a > 0.3 and t > 0.3:
        synth = "Rock/Metal"
    # Electronic/Dance or Jazz (with drums)
    elif (drums_p > 0.2 or cymbals_p > 0.15) and a > 0.3:
        if sax_p > 0.1 or trumpet_p > 0.1:
            synth = "Jazz/Big Band"
        elif synth_p > 0.1:
            synth = "Electronic/Dance"
        else:
            synth = "Rhythmic/Upbeat"
    elif has("epic"):
        synth = "Epic/Cinematic"
    elif has("horror") or (has("dark") and has("heavy")):
        synth = "Dark/Horror"
    elif has("heavy") or has("metal"):
        synth = "Heavy/Metal"
    elif has("relaxing") and has("soft"):
        synth = "Ambient/Chill"
    elif has("meditative") or has("soundscape") or has("ambiental"):
        synth = "Ambient/Textural"
    elif has("party") or has("groovy"):
        synth = "Dance/Funky"
    elif has("upbeat") and a > 0.2:
        synth = "Uplifting/Energetic"
    elif has("romantic") or has("love"):
        synth = "Romantic/Ballad"
    elif has("sad") or has("melancholic"):
        synth = "Melancholic/Sad"
    elif has("film") or has("movie"):
        synth = "Cinematic/Soundtrack"
    elif has("action") and a > 0.3:
        synth = "Action/Adventure"
    elif has("sport") or has("game"):
        synth = "Sport/Game"
    elif has("trailer"):
        synth = "Trailer/Cinematic"
    elif has("fast"):
        synth = "Fast/Hyper"
    elif has("retro"):
        synth = "Retro/Electronic"
    elif has("children") or has("christmas"):
        synth = "Holiday/Family"
    elif has("funny") or has("fun"):
        synth = "Playful/Fun"
    elif has("documentary") or has("corporate"):
        synth = "Ambient/Background"
    elif a > 0.5 and t > 0.3:
        synth = "Intense/Dramatic"
    elif a < 0.1 and t < 0.1:
        synth = "Calm/Peaceful"
    else:
        # Fall back to genre head
        genre_probs = torch.softmax(out["genre_logits"].cpu(), dim=1)
        avg_probs = genre_probs.mean(dim=0)
        top_genre = GENRES_20[avg_probs.topk(1).indices.item()]
        synth = f"{top_genre.capitalize()}"
    
    results["genre_synthesized"] = synth
    results["genres"] = [(synth, 1.0)]

    # Per-segment style (from mood)
    seg_styles = []
    for i in range(len(segments)):
        top = mood_probs[i].topk(1).indices.item()
        seg_styles.append(mood_names[top])
    results["style_curve"] = seg_styles

    # Instrument (detected if prob > 0.4)
    instr_logits = out["instrument_logits"].cpu()
    instr_probs = torch.sigmoid(instr_logits)
    avg_instr = instr_probs.mean(dim=0)
    detected = [(INSTRUMENTS[i], avg_instr[i].item()) for i in range(len(INSTRUMENTS))
                if avg_instr[i] > 0.1]
    results["instruments"] = sorted(detected, key=lambda x: -x[1])
    results["instrument_raw"] = {INSTRUMENTS[i]: avg_instr[i].item() for i in range(len(INSTRUMENTS))}

    results["times"] = times_min
    results["duration"] = total_sec

    return results


def print_results(results, name):
    if results is None:
        print("  (no data)")
        return

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Duration: {results['duration']:.1f}s ({results['duration']/60:.1f} min)")
    print(f"{'='*60}")

    # Emotion
    print(f"\n  --- EMOTION:")
    for dim in ["V", "A", "T", "C"]:
        val = results[f"{dim}_mean"]
        bar = "*" * int(abs(val) * 10) + "-" * (10 - int(abs(val) * 10))
        print(f"    {dim}: {val:>+6.3f}")

    # Style (from mood - primary genre/style descriptor)
    print(f"\n  --- STYLE:")
    synth = results.get("genre_synthesized", "")
    if synth:
        print(f"    >> {synth} <<")
    for m, p in results.get("moods", []):
        bar = "*" * int(p * 20)
        print(f"    {m:20s}: {p:.1%}  {bar}")

    # Instrument
    print(f"\n  --- INSTRUMENT:")
    for inst, p in results["instruments"]:
        bar = "*" * int(p * 20)
        print(f"    {inst:20s}: {p:.0%}  {bar}")

    # Timeline
    print(f"\n  --- TIMELINE (15s windows):")
    print(f"  {'Time':>8} | {'Mood':>14} | V     A     T     C")
    print(f"  {'-'*8}-+-{'-'*14}-+{'-'*22}")
    for i in range(len(results["times"])):
        tm = results["times"][i]
        seg_m = results["style_curve"][i]
        v = results["V_curve"][i]
        a = results["A_curve"][i]
        t = results["T_curve"][i]
        c = results["C_curve"][i]
        print(f"  {tm:>5.1f}m  | {seg_m:>14} | {v:+.2f} {a:+.2f} {t:+.2f} {c:+.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python infer.py test_audio/colosseum_bgm.mp3")
        print("  python infer.py test_audio/sad_piano.mp3")
        print("  python infer.py test_audio/tense_bgm.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print(f"Loading model...")
    model = load_model()

    print(f"Analyzing: {audio_path}")
    results = analyze(model, audio_path)

    name = os.path.splitext(os.path.basename(audio_path))[0]
    print_results(results, name)

    # Save JSON
    json_path = os.path.splitext(audio_path)[0] + "_analysis.json"
    # Convert tensors for JSON
    clean = {k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")
