# Troubleshooting

Real issues encountered while building this pipeline, and how they were resolved.

## 1. RISC-V compile fails on `onnx.LayerNormalization`

**Symptom:** `iree-compile` fails when cross-compiling for RISC-V at the default
optimization level, with an error referencing `onnx.LayerNormalization`.

**Cause:** Aggressive optimization generated vectorized IR that the current LLVM/IREE
RISC-V backend could not lower. This is likely a limitation of the current IREE RISC-V
backend for transformer-style workloads rather than a bug in the model export.

**Solution:** Compile with a lower optimization level:

```bash
--iree-opt-level=O0
```

**Result:** Successful compilation and execution on RISC-V.

## 2. `ImportError: cannot import name DTensor`

**Cause:** `transformers` version incompatible with the installed PyTorch version.

**Solution:** Install compatible versions of `torch` and `transformers`
(see `requirements.txt`).

## 3. Shell shows `import-im6.q16` or similar garbage

**Cause:** Python code was typed directly into the Linux shell instead of a Python
interpreter or script.

**Incorrect:**

```bash
import numpy as np
```

**Correct:** run `python` first, or save the code to a `.py` file and execute it:

```bash
python
# or
python scripts/export_gpt2.py
```

## 4. `iree-run-module`: `expected 1 arguments but passed 0`

**Cause:** No input tensor was supplied.

**Correct:**

```bash
--input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

## 5. Huge output printed to the terminal

This is expected — GPT-2 returns 25 tensors (see
[`compilation_notes.md`](compilation_notes.md)), and the logits tensor alone contains
over 800,000 floating-point values. Redirect to a log file and inspect with `grep`/`head`
if you only need to confirm success:

```bash
iree-run-module ... > logs/run.log 2>&1
grep -E 'EXEC @main_graph' logs/run.log
```

## 6. `Exec format error` running the RISC-V binary

**Cause:** A RISC-V ELF (`iree-run-module`) was launched directly on an x86 host.

**Solution:** Use the QEMU wrapper (`iree-run-module-riscv64`) or invoke
`qemu-riscv64` explicitly — never run the RISC-V binary directly on x86.

## 7. VMFB reports `embedded-elf-x86_64` when a RISC-V artifact was expected

**Cause:** The model was compiled for the host target, not RISC-V.

**Solution:** Re-run the RISC-V compile command with
`--iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d`.
