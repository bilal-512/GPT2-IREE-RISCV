# RISC-V Cross-Compilation and QEMU Execution

This doc covers cross-compiling the imported GPT-2 MLIR for RISC-V (RV64GC) and running
the resulting VMFB under QEMU user-mode emulation.

## Target

- ISA: RISC-V RV64GC
- ABI: `lp64d`
- CPU features: `+m,+a,+f,+d,+c`
- Emulator: `qemu-riscv64`
- Runtime: RISC-V build of `iree-run-module`

## Step 1 — Attempt default-optimization compile (fails)

Compiling with the default optimization level fails while lowering `onnx.LayerNormalization`:

```bash
iree-compile \
  mlir/gpt2.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu=generic-rv64 \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c" \
  -o vmfb/gpt2_riscv64.vmfb
```

Compiler error references `onnx.LayerNormalization`: the aggressive optimizer generated
vectorized IR that the current LLVM/IREE RISC-V backend could not lower.

## Step 2 — Fix: drop to `--iree-opt-level=O0`

```bash
iree-compile \
  mlir/gpt2.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu=generic-rv64 \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c" \
  --iree-opt-level=O0 \
  -o vmfb/gpt2_riscv64.vmfb
```

Output: `vmfb/gpt2_riscv64.vmfb`

## Step 3 — Verify the RISC-V VMFB

```bash
iree-dump-module \
  --output=metadata \
  vmfb/gpt2_riscv64.vmfb
```

Expected:

```
embedded-elf-riscv_64

main_graph
main_graph$async
```

## Step 4 — Verify the RISC-V runtime

```bash
which iree-run-module-riscv64
# /usr/local/bin/iree-run-module-riscv64

iree-run-module-riscv64 --list_drivers
# local-sync
# local-task
```

> Do not run the RISC-V `iree-run-module` binary directly on an x86 host — it is a
> RISC-V ELF and will fail with `Exec format error`. Always go through the QEMU
> wrapper (`iree-run-module-riscv64`) or `qemu-riscv64` explicitly.

## Step 5 — Execute GPT-2 on RISC-V under QEMU

```bash
time iree-run-module-riscv64 \
  --device=local-sync \
  --module=vmfb/gpt2_riscv64.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

Result: successful execution, output logits generated.

Execution time:

```
real   44.292 s
user   42.994 s
sys     1.292 s
```

> QEMU measurements validate functional correctness and relative behavior in this
> environment only. They are emulation timings and must **not** be reported as native
> RISC-V hardware performance. See [`benchmark.md`](benchmark.md).

## Current limitation

The model currently performs forward inference only — a single call to `main_graph`
producing logits (+ KV cache) for a fixed 16-token input. No autoregressive decoding
loop has been implemented yet. See the Roadmap in the top-level README.
