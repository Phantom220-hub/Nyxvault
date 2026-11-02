#!/usr/bin/env python3
"""Export a checkpoint to ONNX; dynamic INT8 quantization is optional."""
import argparse, torch
from pathlib import Path
from model import NYXVAULTGPT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/best.pt"); p.add_argument("--out",default="mobile/nyxvault.onnx"); p.add_argument("--quantize",action="store_true"); a=p.parse_args()
 ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); m=NYXVAULTGPT(ck["vocab_size"],ck["config"]); m.load_state_dict(ck["model"]); m.eval()
 class Export(torch.nn.Module):
  def __init__(self,model): super().__init__(); self.model=model
  def forward(self,input_ids): return self.model(input_ids)[0]
 target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True)
 torch.onnx.export(Export(m),torch.ones((1,16),dtype=torch.long),str(target),input_names=["input_ids"],output_names=["logits"],dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"logits":{0:"batch",1:"sequence"}},opset_version=17)
 if a.quantize:
  from onnxruntime.quantization import quantize_dynamic,QuantType
  q=target.with_name(target.stem+"_int8.onnx"); quantize_dynamic(str(target),str(q),weight_type=QuantType.QInt8); print("Quantized:",q)
 print("Exported:",target,"size_MB",round(target.stat().st_size/1e6,2))
if __name__=="__main__": main()
