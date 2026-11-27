"""Tiny character-level neural language model trained from random weights."""
import json
from pathlib import Path
import numpy as np

DATA = Path("data/knowledge.jsonl")
MODEL = Path("data/neural_model.npz")
CONTEXT, EMBED, HIDDEN, EPOCHS, BATCH = 6, 12, 48, 18, 128

def train_neural(data_path=DATA, model_path=MODEL):
    parts = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("question") and row.get("answer"):
                parts.append("Вопрос: " + row["question"] + "\nОтвет: " + row["answer"] + "\n")
        except (ValueError, TypeError, AttributeError):
            pass
    corpus = "\n".join(parts)
    if len(corpus) < CONTEXT + 2:
        raise ValueError("Недостаточно обучающего текста.")
    chars = sorted(set(corpus))
    to_id = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([to_id[ch] for ch in corpus], dtype=np.int32)
    x = np.zeros((len(ids)-1, CONTEXT), dtype=np.int32)
    for i in range(len(ids)-1):
        seq = ids[max(0, i-CONTEXT+1):i+1]
        x[i, -len(seq):] = seq
    y = ids[1:]
    rng = np.random.default_rng(20261101)
    v = len(chars)
    emb = rng.normal(0, .08, (v, EMBED)).astype(np.float32)
    w1 = rng.normal(0, .08, (CONTEXT*EMBED, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    w2 = rng.normal(0, .05, (HIDDEN, v)).astype(np.float32)
    b2 = np.zeros(v, np.float32)
    loss_value = 0.0
    for epoch in range(EPOCHS):
        order = rng.permutation(len(y))
        total, count = 0.0, 0
        for start in range(0, len(order), BATCH):
            ix = order[start:start+BATCH]
            xb, yb = x[ix], y[ix]
            e = emb[xb].reshape(len(ix), -1)
            h = np.tanh(e @ w1 + b1)
            logits = h @ w2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            loss = -np.log(p[np.arange(len(ix)), yb] + 1e-9).mean()
            total += float(loss)*len(ix); count += len(ix)
            dz = p
            dz[np.arange(len(ix)), yb] -= 1
            dz /= len(ix)
            dw2, db2g = h.T @ dz, dz.sum(axis=0)
            dh = (dz @ w2.T) * (1-h*h)
            dw1, db1g = e.T @ dh, dh.sum(axis=0)
            de = (dh @ w1.T).reshape(len(ix), CONTEXT, EMBED)
            de_table = np.zeros_like(emb)
            np.add.at(de_table, xb.reshape(-1), de.reshape(-1, EMBED))
            for param, grad in ((emb,de_table),(w1,dw1),(b1,db1g),(w2,dw2),(b2,db2g)):
                np.clip(grad, -1, 1, out=grad)
                param -= .035 * grad
        loss_value = total/max(1,count)
    model_path = Path(model_path); model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, chars=np.array(chars), emb=emb, w1=w1, b1=b1,
                         w2=w2, b2=b2, context=np.array([CONTEXT], dtype=np.int32))
    return {"characters":len(corpus),"vocabulary":v,"epochs":EPOCHS,
            "loss":round(loss_value,4),"model":str(model_path)}

class NeuralBrain:
    def __init__(self, model_path=MODEL):
        self.data = np.load(model_path, allow_pickle=False)
        self.chars = self.data["chars"].tolist()
        self.lookup = {c:i for i,c in enumerate(self.chars)}
        self.context = int(self.data["context"][0])

    def generate(self, prompt="Ответ:", max_new=120, temperature=.8):
        ids = [self.lookup.get(c,0) for c in prompt]
        rng = np.random.default_rng()
        for _ in range(max_new):
            ctx = ([0]*self.context + ids)[-self.context:]
            h = np.tanh(self.data["emb"][np.array(ctx)].reshape(1,-1) @ self.data["w1"] + self.data["b1"])
            z = (h @ self.data["w2"] + self.data["b2"]).ravel()/max(.15,temperature)
            z -= z.max(); p = np.exp(z); p /= p.sum()
            nxt = int(rng.choice(len(self.chars),p=p)); ids.append(nxt)
            if self.chars[nxt] == "\n": break
        return "".join(self.chars[i] for i in ids)[len(prompt):].strip()
