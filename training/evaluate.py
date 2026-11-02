#!/usr/bin/env python3
"""Compute validation loss/perplexity and generate 30 fixed smoke-test prompts."""
import argparse, json, math
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import NYXVAULTGPT
PROMPTS=[
"Объясни, что такое алгоритм.", "Что такое переменная в Python?", "Чем список отличается от кортежа?",
"Объясни цикл for на простом примере.", "Что такое функция?", "Что такое рекурсия?",
"Для чего нужен Git?", "Что такое GitHub?", "Что такое API?", "Объясни HTTP простыми словами.",
"Что такое база данных?", "Чем SQL отличается от NoSQL?", "Что такое операционная система?",
"Как работает компилятор?", "Что такое C++?", "Что такое Python?", "Что такое машинное обучение?",
"Чем нейросеть отличается от обычной программы?", "Что такое токенизатор BPE?", "Что такое Transformer?",
"Как устроена видеоигра?", "Что такое Godot Engine?", "Что такое Android APK?",
"Что такое кибербезопасность?", "Объясни облачные вычисления.", "Интернеттің маңызы туралы эссе жаз.",
"Эссе дар бораи Ватан навис.", "Дар бораи забони тоҷикӣ иншо навис.",
"Табиатро ҳифз кардан чаро муҳим аст?", "Илм ва технология дар ҳаёти ҷавонон."
]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/best.pt"); p.add_argument("--tokenizer",default="artifacts/tokenizer.json"); p.add_argument("--data",default="training_data/validation.txt"); p.add_argument("--max-new",type=int,default=100); a=p.parse_args()
 dev="cuda" if torch.cuda.is_available() else "cpu"; ck=torch.load(a.checkpoint,map_location=dev,weights_only=False)
 tok=Tokenizer.from_file(a.tokenizer); model=NYXVAULTGPT(ck["vocab_size"],ck["config"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
 ids=tok.encode(Path(a.data).read_text(encoding="utf-8")).ids; x=torch.tensor(ids,dtype=torch.long,device=dev)
 with torch.no_grad():
  chunks=[]
  for i in range(0,len(x)-model.cfg["block_size"]-1,model.cfg["block_size"]):
   z=x[i:i+model.cfg["block_size"]].unsqueeze(0); y=x[i+1:i+model.cfg["block_size"]+1].unsqueeze(0); chunks.append(model(z,y)[1].item())
 loss=sum(chunks)/max(1,len(chunks)); print(json.dumps({"validation_loss":loss,"perplexity":math.exp(min(loss,30)),"evaluated_chunks":len(chunks)},indent=2))
 for prompt in PROMPTS:
  inp=tok.encode(prompt).ids[-model.cfg["block_size"]:]
  t=torch.tensor([inp],device=dev)
  with torch.no_grad():
   for _ in range(a.max_new):
    logits,_=model(t); nxt=torch.multinomial(torch.softmax(logits[0,-1]/0.85,dim=-1),1).view(1,1); t=torch.cat([t,nxt],dim=1)
  print("\nPROMPT:",prompt,"\nOUTPUT:",tok.decode(t[0].tolist())[len(tok.decode(inp)):].strip())
 print("\nОценка 30 ответов требует ручной проверки: связность, фактическая точность, следование теме. PPL не измеряет истинность.")
if __name__=="__main__": main()
