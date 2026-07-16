# Compilation Notes — MLIR / IREE Internals

## Why does GPT-2 produce 25 outputs?

The compiled `main_graph` function returns **25 tensors**, not just logits. This is expected
and is part of GPT-2's caching mechanism for autoregressive generation:

```
1  logits tensor
+
24 past_key_values tensors   (12 transformer layers × [key, value])
=
25 outputs
```

GPT-2 Small has 12 transformer layers. Each layer returns a key cache and a value cache,
so `12 × 2 = 24` KV-cache tensors, plus the single logits tensor, gives 25 outputs total.

## Expected shapes

| Tensor | Shape |
|---|---|
| Input (`input_ids`) | `1 × 16 × int64` |
| Output (`logits`) | `1 × 16 × 50257` |
| Additional outputs | 24 × past key/value tensors |

The logits tensor alone contains over 800,000 floating-point values — the large console
output when running `iree-run-module` is expected, not an error.

## IREE Compilation Pipeline

`iree-compile` lowers the imported MLIR through a sequence of internal phases before
emitting the final VMFB:

```
Input MLIR
    │
    ▼
ABI Generation
    │
    ▼
Preprocessing
    │
    ▼
Global Optimizations
    │
    ▼
Dispatch Creation
    │
    ▼
Flow
    │
    ▼
Stream
    │
    ▼
Executable Sources
    │
    ▼
Executable Configurations
    │
    ▼
Executable Targets
    │
    ▼
HAL
    │
    ▼
VM
    │
    ▼
VMFB
```

| Phase | Purpose |
|---|---|
| ABI | Public function signatures adapted to the IREE runtime ABI / buffer-view interface |
| Preprocessing | Canonicalization and frontend-specific cleanup |
| Global optimization | Whole-program tensor and operation optimizations |
| Dispatch creation | Computational regions grouped into executable dispatches |
| Flow | Dispatch-level dataflow and workload structure |
| Stream | Resource lifetimes, command ordering, async execution scheduling |
| Executable sources/configs/targets | Target-specific lowering and code-gen configuration |
| HAL | Hardware abstraction layer — devices, buffers, executables, commands |
| VM | Final IREE VM program before serialization to VMFB |

## Backend lowering issue (RISC-V)

At the default optimization level, `onnx.LayerNormalization` was lowered into vectorized
IR that the current LLVM/IREE RISC-V backend could not handle. See
[`troubleshooting.md`](troubleshooting.md) and [`riscv_execution.md`](riscv_execution.md)
for the fix (`--iree-opt-level=O0`).
