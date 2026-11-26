#!/usr/bin/env python3
"""Compute validation loss/perplexity and generate 30 fixed smoke-test prompts."""
import argparse, json, math
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import NYXVAULTGPT
PROMPTS=[
"Объясни переменные, типы int, float, str и bool в Python.",
"Напиши функцию Python, которая возвращает квадрат числа.",
"Напиши функцию Python, которая проверяет, является ли число чётным.",
"Объясни разницу между for и while в Python.",
"Как добавить элемент в список Python?",
"Объясни dict и покажи пример словаря.",
"Найди ошибку в коде Python: if x = 5: print(x).",
"Напиши пример try/except для обработки ValueError.",
"Что такое функция, параметр и return?",
"Объясни рекурсию на небольшом примере.",
"Покажи минимальный HTML-документ с заголовком и абзацем.",
"Как создать ссылку и изображение в HTML с alt?",
"Объясни CSS box model.",
"Как расположить элементы по центру с Flexbox?",
"В чём разница между CSS Grid и Flexbox?",
"Напиши media query для узкого экрана.",
"Чем let отличается от const в JavaScript?",
"Напиши обработчик события click для кнопки.",
"Как map() отличается от forEach() в JavaScript?",
"Покажи обработку ошибки в async fetch.",
"Объясни JSON.parse и JSON.stringify.",
"Что такое DOM и querySelector?",
"Как создать Git commit из изменений?",
"Чем git pull отличается от git push?",
"Что такое ветка и pull request на GitHub?",
"Что такое алгоритмическая сложность O(n)?",
"Объясни стек и очередь данных.",
"Как перемещать CharacterBody2D в Godot 4?",
"Что такое _ready, _process и _physics_process в Godot?",
"Как работают сигналы в Godot?",
"Напиши функцию GDScript для вычисления суммы двух чисел.",
"Что такое Kotlin и где он применяется?",
"Для чего в C++ используют vector?",
"Как устроен простой unit-тест?",
"Чаро санҷидани барнома бо тестҳо муҳим аст?"
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
