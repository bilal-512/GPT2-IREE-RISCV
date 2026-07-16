# Deployment Pipeline

Full, reproducible walkthrough of exporting GPT-2 Small from PyTorch and running it through
the IREE compiler stack on both x86 and RISC-V.

## Objective

- Export GPT-2 Small from PyTorch to ONNX
- Verify correctness with ONNX Runtime
- Convert the ONNX model into MLIR
- Compile MLIR into an optimized VMFB executable
- Execute the compiled model with the IREE Runtime (x86 and RISC-V/QEMU)
- Understand the internal MLIR compilation pipeline

## Requirements

- Ubuntu 22.04 (recommended)
- Python >= 3.10
- IREE tools: `iree-import-onnx`, `iree-compile`, `iree-run-module`
- Python packages: `torch`, `transformers`, `onnx`, `onnxruntime`, `numpy`

## Project Structure (working directory)

```
GPT2_IREE/
├── .venv/
├── model/
├── mlir/
├── vmfb/
├── logs/
├── scripts/
│   └── export_gpt2.py
└── test_onnx.py
```

## Step 1 — Create the Project

```bash
mkdir GPT2_IREE
cd GPT2_IREE

mkdir model mlir vmfb logs scripts
```

## Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install Required Packages

CPU-only setup:

```bash
pip install torch transformers onnx onnxruntime numpy
```

CUDA packages are not required unless GPU acceleration is desired.

## Step 4 — Export GPT-2 to ONNX

`scripts/export_gpt2.py`:

```python
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

dummy = torch.randint(0, 1000, (1, 16), dtype=torch.long)

torch.onnx.export(
    model,
    dummy,
    "model/gpt2.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "input_ids": {1: "sequence"},
        "logits": {1: "sequence"}
    }
)

print("Export complete")
```

Run:

```bash
python scripts/export_gpt2.py
```

Expected output: `Export complete`

## Step 5 — Verify the ONNX Model

```python
import onnx

m = onnx.load("model/gpt2.onnx")
onnx.checker.check_model(m)

print("ONNX OK")
```

Expected output: `ONNX OK`

## Step 6 — Test with ONNX Runtime

`test_onnx.py`:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("model/gpt2.onnx")

x = np.random.randint(0, 1000, (1, 16), dtype=np.int64)

y = sess.run(None, {"input_ids": x})

print("Output shape:", y[0].shape)
```

Run:

```bash
python test_onnx.py
```

Expected output: `Output shape: (1, 16, 50257)`

## Step 7 — Import ONNX → MLIR

```bash
iree-import-onnx \
  model/gpt2.onnx \
  -o mlir/gpt2.mlir
```

Verify:

```bash
ls mlir
# gpt2.mlir
```

## Step 8 — Inspect the MLIR Signature

```bash
grep -n "func.func @main_graph" mlir/gpt2.mlir
```

Expected (similar to):

```
func.func @main_graph(
    input_ids
) -> 25 outputs
```

See [`compilation_notes.md`](compilation_notes.md) for why there are 25 outputs.

## Step 9 — Compile MLIR → VMFB (x86)

```bash
iree-compile \
  mlir/gpt2.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  -o vmfb/gpt2.vmfb
```

Verify:

```bash
ls vmfb
# gpt2.vmfb
```

## Step 10 — Execute on x86 with IREE

```bash
iree-run-module \
  --device=local-task \
  --module=vmfb/gpt2.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

Expected output: `EXEC @main_graph` followed by a large tensor output (logits + KV cache).

## Step 11 — Cross-Compile and Run for RISC-V

See [`riscv_execution.md`](riscv_execution.md) for the RISC-V-specific compile flags,
the `--iree-opt-level=O0` workaround, and QEMU execution.

## Complete Command Sequence

```bash
mkdir GPT2_IREE
cd GPT2_IREE

mkdir model mlir vmfb logs scripts

python3 -m venv .venv
source .venv/bin/activate

pip install torch transformers onnx onnxruntime numpy

python scripts/export_gpt2.py
python test_onnx.py

iree-import-onnx model/gpt2.onnx -o mlir/gpt2.mlir

iree-compile \
  mlir/gpt2.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  -o vmfb/gpt2.vmfb

grep -n "func.func @main_graph" mlir/gpt2.mlir

iree-run-module \
  --device=local-task \
  --module=vmfb/gpt2.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```
