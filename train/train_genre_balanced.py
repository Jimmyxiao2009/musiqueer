"""
Balanced genre head training
"""
import sys, os, torch, yaml, glob, random, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import librosa
from torch.utils.data import Dataset, DataLoader
from model.music_understand import MusicUnderstand2
from utils.audio import mel_spectrogram, aggregate_frames
from data.mtg_jamendo import load_metadata as load_mtg, CACHE_DIR as MTG_CACHE

GENRES = ['blues','classical','country','disco','electronic','folk','hiphop','jazz',
    'metal','pop','reggae','rock','instrumental','international','soul_rnb',
    'experimental','spoken','ambient','soundtrack','orchestral']
NUM_GENRES = len(GENRES); G2I = {g:i for i,g in enumerate(GENRES)}

GTZAN = os.path.join(os.path.expanduser("~"),
    '.cache/kagglehub/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original')
GTZAN_MAP = {'blues':'blues','classical':'classical','country':'country','disco':'disco',
    'hiphop':'hiphop','jazz':'jazz','metal':'metal','pop':'pop','reggae':'reggae','rock':'rock'}
FMA_MAP = {'Rock':'rock','Experimental':'experimental','Electronic':'electronic','Hip-Hop':'hiphop',
    'Folk':'folk','Pop':'pop','Instrumental':'instrumental','International':'international',
    'Classical':'classical','Jazz':'jazz','Old-Time / Historic':'folk','Spoken':'spoken',
    'Country':'country','Soul-RnB':'soul_rnb','Blues':'blues','Easy Listening':'ambient'}
FMA_CACHE = 'data/fma_cache'

# Build samples
samples = []
for gtzan_g, exp_g in GTZAN_MAP.items():
    path = os.path.join(GTZAN, gtzan_g)
    if not os.path.isdir(path): continue
    files = glob.glob(os.path.join(path, '*.wav'))
    random.Random(42).shuffle(files)
    for f in files[:200]:
        samples.append((f, G2I[exp_g], 'gtzan'))

fma_meta = pd.read_csv('data/fma_meta/tracks.csv', header=[0,1], low_memory=False)
fma_meta.columns = ['track_id' if 'Unnamed' in str(a) and 'Unnamed' in str(b)
    else f'{a}_{b}' if b and 'Unnamed' not in str(b) else a for a,b in fma_meta.columns]
for fname in os.listdir(FMA_CACHE):
    if not fname.endswith('.pt'): continue
    tid = int(fname[:-3])
    row = fma_meta[fma_meta['track_id']==tid]
    if len(row)==0: continue
    g = row.iloc[0].get('track_genre_top', None)
    if pd.notna(g) and str(g) in FMA_MAP:
        samples.append((os.path.join(FMA_CACHE, fname), G2I[FMA_MAP[str(g)]], 'fma'))

print(f'Total: {len(samples)}')
gc = {}
for _,l,_ in samples: gc[GENRES[l]] = gc.get(GENRES[l],0)+1
for g,c in sorted(gc.items(),key=lambda x:-x[1]):
    print(f'  {g:15s}: {c}')

random.Random(42).shuffle(samples)
n_val = int(len(samples)*0.15)

class GDS(Dataset):
    def __init__(self, xs): self.xs = xs
    def __len__(self): return len(self.xs)
    def __getitem__(self, idx):
        path, lbl, src = self.xs[idx]
        if src == 'fma':
            d = torch.load(path, map_location='cpu', weights_only=True); mel = d['mel']
        else:
            try: y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
            except: return self[(idx+1)%len(self)]
            yt = torch.from_numpy(y).unsqueeze(0).float()
            tl = 30*22050
            if yt.shape[-1] < tl: yt = torch.nn.functional.pad(yt, (0, tl-yt.shape[-1]))
            mel = mel_spectrogram(yt, 22050, n_mels=128); mel = aggregate_frames(mel, 21)
        T = mel.shape[-1]
        if T > 61: mel = mel[...,:61]
        elif T < 61: mel = torch.nn.functional.pad(mel, (0, 61-T))
        return mel, lbl

with open('config/default.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = MusicUnderstand2(backbone_config=cfg['model']['backbone'],
    emotion_config={k:v for k,v in cfg['model']['experts']['emotion'].items() if k!='loss_weight'},
    mood_config={'num_tags':59}, genre_config={'num_genres':NUM_GENRES})
ckpt = torch.load('checkpoints/joint_latest.pth', map_location='cpu', weights_only=True)
sd = ckpt['model_state_dict']
for k in list(sd.keys()):
    if k in model.state_dict() and sd[k].shape != model.state_dict()[k].shape: del sd[k]
model.load_state_dict(sd, strict=False)
for p in model.parameters(): p.requires_grad = False
for n,p in model.named_parameters():
    if 'genre_head' in n: p.requires_grad = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
tl = DataLoader(GDS(samples[n_val:]), batch_size=32, shuffle=True)
vl = DataLoader(GDS(samples[:n_val]), batch_size=32)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

best_acc = 0
for epoch in range(1, 31):
    model.train()
    for mel, lbl in tl:
        mel, lbl = mel.to(device), lbl.to(device)
        logits = model(mel, active_experts=['genre'])['genre_logits']
        torch.nn.functional.cross_entropy(logits, lbl).backward()
        opt.step(); opt.zero_grad()
    
    model.eval()
    nc, nt = 0, 0
    with torch.no_grad():
        for mel, lbl in vl:
            logits = model(mel.to(device), active_experts=['genre'])['genre_logits']
            nc += (logits.argmax(dim=1).cpu()==lbl).sum().item(); nt += len(lbl)
    acc = nc/nt
    if acc > best_acc: best_acc = acc
    if epoch%5==0: print(f'Epoch {epoch:2d}: val_acc={acc:.3f}')

torch.save({'model_state_dict': model.state_dict()},'checkpoints/genre_balanced.pth')
print(f'\nDone! Best val_acc={best_acc:.3f}')
