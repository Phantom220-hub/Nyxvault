"""Small GPT-style decoder-only Transformer; initialized randomly (no pretrained weights)."""
import torch
from torch import nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model); self.proj=nn.Linear(d_model,d_model)
        self.attn_drop=nn.Dropout(dropout); self.resid_drop=nn.Dropout(dropout)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size),persistent=False)
    def forward(self,x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        def split(z): return z.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k,v=map(split,(q,k,v))
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=self.mask[:,:,:t,:t],dropout_p=self.attn_drop.p if self.training else 0.0)
        y=y.transpose(1,2).contiguous().view(b,t,c)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg["d_model"]); self.attn=CausalSelfAttention(cfg["d_model"],cfg["n_heads"],cfg["dropout"],cfg["block_size"])
        self.ln2=nn.LayerNorm(cfg["d_model"])
        self.mlp=nn.Sequential(nn.Linear(cfg["d_model"],4*cfg["d_model"]),nn.GELU(),nn.Linear(4*cfg["d_model"],cfg["d_model"]),nn.Dropout(cfg["dropout"]))
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))

class NYXVAULTGPT(nn.Module):
    def __init__(self,vocab_size,cfg=None):
        super().__init__()
        self.cfg=cfg or {"d_model":384,"n_heads":8,"n_layers":8,"block_size":256,"dropout":0.1}
        c=self.cfg
        self.token_emb=nn.Embedding(vocab_size,c["d_model"]); self.pos_emb=nn.Embedding(c["block_size"],c["d_model"])
        self.drop=nn.Dropout(c["dropout"]); self.blocks=nn.Sequential(*[Block(c) for _ in range(c["n_layers"])])
        self.ln_f=nn.LayerNorm(c["d_model"]); self.head=nn.Linear(c["d_model"],vocab_size,bias=False)
        self.head.weight=self.token_emb.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,.02)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.cfg["block_size"]: idx=idx[:,-self.cfg["block_size"]:]; t=idx.size(1)
        pos=torch.arange(t,device=idx.device)
        x=self.drop(self.token_emb(idx)+self.pos_emb(pos)[None,:,:])
        logits=self.head(self.ln_f(self.blocks(x)))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets[:,-t:].reshape(-1)) if targets is not None else None
        return logits,loss
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
