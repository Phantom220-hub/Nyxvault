#!/usr/bin/env python3
"""Clean and split licensed/user-owned text into train/validation corpora."""
import argparse, json, random, re
from pathlib import Path

def clean(s):
    s=s.replace("\x00","").replace("\r","\n")
    s=re.sub(r"https?://\S+"," ",s)
    s=re.sub(r"[ \t]+"," ",s)
    s=re.sub(r"\n{3,}","\n\n",s)
    return s.strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="training_data/raw")
    p.add_argument("--knowledge",default="data/knowledge.jsonl")
    p.add_argument("--out",default="training_data")
    p.add_argument("--val-ratio",type=float,default=.05)
    p.add_argument("--min-chars",type=int,default=80)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); root=Path(a.input); docs=[]
    if root.exists():
        for f in root.rglob("*.txt"):
            t=clean(f.read_text(encoding="utf-8",errors="ignore"))
            if len(t)>=a.min_chars: docs.append(t)
        for f in root.rglob("*.jsonl"):
            for line in f.read_text(encoding="utf-8",errors="ignore").splitlines():
                try:
                    x=json.loads(line)
                    t=clean("Вопрос: "+str(x.get("question",""))+"\nОтвет: "+str(x.get("answer","")))
                    if len(t)>=20: docs.append(t)
                except Exception: pass
    k=Path(a.knowledge)
    if k.exists():
        for line in k.read_text(encoding="utf-8",errors="ignore").splitlines():
            try:
                x=json.loads(line); t=clean("Вопрос: "+str(x.get("question",""))+"\nОтвет: "+str(x.get("answer","")))
                if len(t)>=20: docs.append(t)
            except Exception: continue
    # Remove exact duplicates without changing document order.
    docs=list(dict.fromkeys(docs))
    if len(docs)<10: raise SystemExit(f"Only {len(docs)} usable documents. Add licensed corpus; refusing tiny split.")
    random.Random(a.seed).shuffle(docs)
    n=max(1,int(len(docs)*a.val_ratio)); val=docs[:n]; train=docs[n:]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"train.txt").write_text("\n\n".join(train),encoding="utf-8")
    (out/"validation.txt").write_text("\n\n".join(val),encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"documents":len(docs),"train_documents":len(train),"validation_documents":len(val),"seed":a.seed},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared {len(docs)} docs: train={len(train)}, validation={len(val)}")
if __name__=="__main__": main()
