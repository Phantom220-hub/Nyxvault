# Third-party code and data licenses

## Training code dependencies
- Python standard library: Python Software Foundation License.
- PyTorch: BSD-style license (see https://github.com/pytorch/pytorch/blob/main/LICENSE).
- tokenizers (Hugging Face): Apache License 2.0 (https://github.com/huggingface/tokenizers/blob/main/LICENSE).
- NumPy: BSD 3-Clause (https://github.com/numpy/numpy/blob/main/LICENSE.txt).
- ONNX: Apache License 2.0 (https://github.com/onnx/onnx/blob/main/LICENSE).
- ONNX Runtime: MIT License (https://github.com/microsoft/onnxruntime/blob/main/LICENSE).
- Kaggle Notebook environment packages are installed from their official PyPI distributions. Exact versions are recorded in the notebook output/requirements file.

No pretrained model weights, model checkpoints, or third-party language-model parameters are downloaded. NYXVAULT parameters are initialized randomly.

## Training data
See [DATA_SOURCES.md](DATA_SOURCES.md). Wikipedia-derived text is generally CC BY-SA 4.0/GFDL subject to page-specific exceptions; attribution metadata is generated per page. Any adapted corpus containing CC BY-SA material must be shared under compatible ShareAlike terms when distributed. Project-owned and public-domain text have separate provenance.
