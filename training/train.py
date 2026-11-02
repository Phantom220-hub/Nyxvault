#!/usr/bin/env python3
"""Train NYXVAULT from random initialization; supports resumable checkpoints."""
import argparse, json, math, random
from pathlib import Path
import numpy as np, torch
from model import NYXVAULTGPT

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--data",default="training_data"); p.add_argument("--tokenizer",default="artifacts/tokenizer.json")
 p.add_argument("--out",default="checkpoints"); p.add_argument("--steps",type=int,default=6000)
 p.add_argument("--batch-size",type=int,default=32); p.add_argument("--block-size",type=int,default=256)
 p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--save-every",type=int,default=500)
 p.add_argument("--resume",default=""); p.add_argument("--seed",type=int,default=1337)
 a=p.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(a.tokenizer); vocab=tok.get_vocab_size()
 def encode(path): return torch.tensor(tok.encode(Path(path).read_text(encoding="utf-8")).ids,dtype=torch.long)
 tr=encode(Path(a.data)/"train.txt"); va=encode(Path(a.data)/"validation.txt")
 if len(tr)<a.block_size+2 or len(va)<a.block_size+2: raise SystemExit("Not enough tokens. Add more licensed text.")
 device="cuda" if torch.cuda.is_available() else "cpu"
 cfg={"d_model":256,"n_heads":8,"n_layers":6,"block_size":a.block_size,"dropout":0.1}
 model=NYXVAULTGPT(vocab,cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1)
 scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda"))
 start=0; best=float("inf")
 if a.resume and Path(a.resume).exists():
  ck=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
  start=ck["step"]+1; best=ck.get("best_val",best)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 def batch(data):
  ix=torch.randint(0,len(data)-a.block_size-1,(a.batch_size,))
  x=torch.stack([data[i:i+a.block_size] for i in ix]); y=torch.stack([data[i+1:i+a.block_size+1] for i in ix])
  return x.to(device),y.to(device)
 @torch.no_grad()
 def estimate():
  model.eval(); vals=[]
  for d in (tr,va):
   ls=[]
   for _ in range(20):
    x,y=batch(d); _,l=model(x,y); ls.append(l.item())
   vals.append(float(np.mean(ls)))
  model.train(); return vals
 for step in range(start,a.steps):
  x,y=batch(tr); opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device=="cuda")):
   _,loss=model(x,y)
  scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
  scaler.step(opt); scaler.update()
  if step%100==0:
   tl,vl=estimate(); print(f"step {step}/{a.steps} train_loss={tl:.4f} val_loss={vl:.4f} val_ppl={math.exp(min(vl,20)):.2f}",flush=True)
   if vl<best:
    best=vl; torch.save({"model":model.state_dict(),"config":cfg,"vocab_size":vocab,"step":step,"best_val":best},out/"best.pt")
  if (step+1)%a.save_every==0 or step+1==a.steps:
   torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"config":cfg,"vocab_size":vocab,"step":step,"best_val":best},out/f"step_{step+1}.pt")
   print("Checkpoint saved",step+1,flush=True)
 print("Training finished. Best validation loss:",best)
if __name__=="__main__": main()
