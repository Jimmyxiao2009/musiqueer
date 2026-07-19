
import os, sys, warnings, json, random
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

BASE = r"H:\musicunderstand2"
sys.path.insert(0, BASE)
os.chdir(BASE)

from model.experts.language_decoder import LanguageDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Tokenizer
tok = Tokenizer.from_file(r"H:\musicunderstand2\tokenizers\gpt2\tokenizer.json")
VOCAB_SIZE = 50257

class SimpleTokenizer:
    def __init__(self, tok):
        self.tok = tok
        self.pad_token_id = 50256
    def encode(self, text):
        return self.tok.encode(text).ids
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

tokenizer = SimpleTokenizer(tok)

# Training texts
with open(r"H:\musicunderstand2\data\merged\training_texts.txt", encoding="utf-8") as f:
    texts = [l.strip() for l in f if l.strip()]
print(f"Texts: {len(texts)}")

# Pure text dataset (no audio)
class TextDataset(Dataset):
    def __init__(self, texts, max_len=64):
        self.texts = texts
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ids = tokenizer.encode(self.texts[idx])
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [tokenizer.pad_token_id] * (self.max_len - len(ids))
        mask = [1 if i != tokenizer.pad_token_id else 0 for i in ids]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

ds = TextDataset(texts)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
print(f"Samples: {len(ds)}, Batches: {len(loader)}")

# Decoder (load best checkpoint)
decoder = LanguageDecoder(d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, vocab_size=VOCAB_SIZE, max_seq_len=128)
best = BASE + r"\checkpoints\language_decoder_final.pth"
if os.path.exists(best):
    decoder.load_state_dict(torch.load(best, map_location=DEVICE, weights_only=True))
    print("Loaded checkpoint")
decoder.to(DEVICE)
print(f"Params: {sum(p.numel() for p in decoder.parameters()):,}")

# Training
optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(10):
    decoder.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Ep {epoch+1}/10")
    for ids, mask in pbar:
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        
        # Create random noise as "memory" (no audio)
        noise_mem = torch.randn(ids.size(0), 30, 256, device=DEVICE)
        logits = decoder(noise_mem, ids, mask)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        ppl = torch.exp(loss).item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "ppl": f"{ppl:.1f}"})
    
    avg_loss = total_loss / len(loader)
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f} ppl={avg_ppl:.1f}")
    
    # Generate sample
    with torch.no_grad():
        decoder.eval()
        noise = torch.randn(1, 30, 256, device=DEVICE)
        input_ids = torch.full((1, 1), 464, dtype=torch.long, device=DEVICE)  # "The"
        attn = torch.ones((1, 1), dtype=torch.long, device=DEVICE)
        for _ in range(32):
            if input_ids.size(1) > 128:
                input_ids = input_ids[:, -128:]
                attn = attn[:, -128:]
            logits = decoder(noise, input_ids, attn)
            next_lg = logits[:, -1, :] / 0.7
            topk_v, topk_i = torch.topk(next_lg, 30, dim=-1)
            next_lg = torch.full_like(next_lg, float("-inf"))
            next_lg.scatter_(-1, topk_i, topk_v)
            probs = torch.softmax(next_lg, dim=-1)
            tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, tok], dim=-1)
            attn = torch.cat([attn, torch.ones((1,1), device=DEVICE, dtype=torch.long)], dim=-1)
            if tok.item() == 50256:
                break
        ids = input_ids[0].tolist()
        if 50256 in ids:
            ids = ids[:ids.index(50256)]
        print(f"  Sample: {tokenizer.decode(ids)}")
        decoder.train()
    
    # Save
    torch.save(decoder.state_dict(), BASE + f"//checkpoints//local_ep{epoch+1}.pth")
    print(f"  Saved ep{epoch+1}.pth")

print("Done!")
