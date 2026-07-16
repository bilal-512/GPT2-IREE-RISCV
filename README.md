# GPT-2 Deployment on RISC-V using IREE

> End-to-end deployment of GPT-2 (124M) from PyTorch to a RISC-V target using the IREE compiler framework, ONNX, MLIR, and QEMU.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![ONNX](https://img.shields.io/badge/ONNX-opset17-green)
![MLIR](https://img.shields.io/badge/MLIR-LLVM-orange)
![IREE](https://img.shields.io/badge/IREE-3.11.0rc-purple)
![RISC--V](https://img.shields.io/badge/RISC--V-RV64GC-red)
![QEMU](https://img.shields.io/badge/QEMU-user--mode-success)
![Status](https://img.shields.io/badge/status-inference%20verified-brightgreen)

---

## Overview

Modern LLMs are almost always deployed on x86 or ARM. This project explores what it takes to cross-compile a transformer model for the open **RISC-V** ISA using Google's **IREE** compiler framework — going all the way from a PyTorch checkpoint down to a RISC-V ELF executed under QEMU.

**Pipeline:**

<img width="1102" height="1201" alt="GPT2_MLIR_FLOW drawio" src="https://github.com/user-attachments/assets/73dc3349-982b-439f-9576-4475fce29df9" />


## Status

| Stage | Status |
|---|---|
| Export GPT-2 (PyTorch → ONNX) | ✅ |
| ONNX verification (`onnx.checker`) | ✅ |
| ONNX Runtime smoke test | ✅ |
| ONNX → MLIR import | ✅ |
| Compile + run on x86 (`local-task`) | ✅ |
| Cross-compile for RISC-V (RV64GC) | ✅ |
| Execute on RISC-V via QEMU | ✅ |
| Correct output logits on both targets | ✅ |
| Autoregressive text generation | ⬜ planned |

## Repository Structure

```
GPT2-IREE-RISCV/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── deployment_pipeline.md   # full step-by-step walkthrough
│   ├── compilation_notes.md     # MLIR / IREE compiler internals, 25-output explanation
│   ├── riscv_execution.md       # RISC-V cross-compile + QEMU execution details
│   ├── benchmark.md             # timings, x86 vs RISC-V/QEMU
│   └── troubleshooting.md       # real issues hit + fixes
│
├── scripts/
│   ├── export_gpt2.py           # PyTorch → ONNX exporter
│   └── compile.sh                # one-shot compile for x86 + RISC-V
│
├── model/                        # gpt2.onnx goes here (not committed, see model/README.md)
├── mlir/                         # gpt2.mlir goes here
├── vmfb/                         # gpt2.vmfb / gpt2_riscv64.vmfb go here
├── logs/                         # run logs (QEMU timing, compiler stderr, etc.)
│
├── test_onnx.py                  # ONNX Runtime smoke test
└── images/                       # diagrams for this README
```

## Quickstart

```bash
git clone https://github.com/bilal-512/GPT2-IREE-RISCV.git
cd GPT2-IREE-RISCV

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Export GPT-2 to ONNX
python scripts/export_gpt2.py

# 2. Sanity-check with ONNX Runtime
python test_onnx.py

# 3. Compile + run for x86 and RISC-V
bash scripts/compile.sh
```

See [`docs/deployment_pipeline.md`](docs/deployment_pipeline.md) for the full walkthrough and [`docs/riscv_execution.md`](docs/riscv_execution.md) for the RISC-V/QEMU-specific steps.

## Technologies

| Component | Technology |
|---|---|
| Model | GPT-2 (124M, 12 layers) |
| Framework | PyTorch, Hugging Face Transformers |
| Interchange format | ONNX (opset 17) |
| IR | MLIR (`main_graph`) |
| Compiler | IREE (`iree-import-onnx`, `iree-compile`) |
| Backend | LLVM CPU |
| Target ISA | RISC-V RV64GC (`lp64d` ABI) |
| Emulation | QEMU user-mode (`qemu-riscv64`) |
| Runtime | IREE Runtime (`iree-run-module`) |

## Results

| Platform | Driver | Status | Output | Time |
|---|---|---|---|---|
| x86 | `local-task` | ✅ | logits `1×16×50257` | not benchmarked in this report |
| RISC-V (QEMU) | `local-sync` | ✅ | logits `1×16×50257` | 44.292 s (real), 42.994 s (user), 1.292 s (sys) |

> GPT-2 exports 25 outputs total: 1 logits tensor + 24 past key/value cache tensors (12 layers × K/V). See [`docs/compilation_notes.md`](docs/compilation_notes.md) for why.

## Interesting Challenge

Compiling for RISC-V with default optimization failed while lowering `onnx.LayerNormalization` — the optimizer generated vectorized IR that the current LLVM/IREE RISC-V backend couldn't lower. Dropping to `--iree-opt-level=O0` resolved it and produced a working RISC-V executable. Full writeup in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Roadmap

- [ ] Tokenizer integration (prompt → input IDs)
- [ ] Greedy decoding loop
- [ ] Top-K / Top-P sampling
- [ ] KV-cache reuse across decode steps
- [ ] Native RISC-V hardware benchmarking (QEMU timings are emulation-only)
- [ ] Performance profiling / opt-level comparison

## Skills Demonstrated

| Area | Skills |
|---|---|
| Machine Learning | GPT-2, Transformers, autoregressive LM internals |
| Model Deployment | ONNX export/verification, IREE |
| Compiler Engineering | MLIR, LLVM, backend lowering & debugging |
| Embedded / Systems AI | RISC-V (RV64GC), QEMU cross-execution |
| Programming | Python, Bash |
| Systems | Linux, cross-compilation toolchains |

## License

MIT — see [LICENSE](LICENSE).
