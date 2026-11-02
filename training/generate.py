#!/usr/bin/env python3
import argparse, torch
from tokenizers import Tokenizer
from model import NYXVAULTGPT
def main():
 p=argparse.ArgumentParser(); p.add_argument("prompt"); p.add_argument("--checkpoint",default="checkpoints/best.pt"); p.add_argument("--tokenizer",default="artifacts/tokenizer.json"); p.add_argument("--max-new",type=int,default=120); p.add_argument("--temperature",type=float,default=.8); a=p.parse_args()
 d="cuda" if torch.cuda.is_available() else "cpu"; ck=torch.load(a.checkpoint,map_location=d,weights_only=False); tok=Tokenizer.from_file(a.tokenizer)
 m=NYXVAULTGPT(ck["vocab_size"],ck["config"]).to(d); m.load_state_dict(ck["model"]); m.eval()
 ids=tok.encode(a.prompt).ids; x=torch.tensor([ids[-m.cfg["block_size"]:]],device=d)
 with torch.no_grad():
  for _ in range(a.max_new):
   logits,_=m(x); probs=torch.softmax(logits[0,-1]/max(a.temperature,.05),-1); nxt=torch.multinomial(probs,1).view(1,1); x=torch.cat([x,nxt],1)
 print(tok.decode(x[0].tolist()))
if __name__=="__main__": main()
