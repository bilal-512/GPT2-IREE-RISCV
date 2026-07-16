# Benchmark

Single forward pass of GPT-2 Small (124M) on a fixed `1×16` `int64` input
(`main_graph`, 25 outputs: logits + 24 KV-cache tensors).

| Platform | Driver | Status | Output | Time |
|---|---|---|---|---|
| x86 (host) | `local-task` | ✅ | `1×16×50257` logits | not benchmarked in this report — measure with `time` on your own host |
| RISC-V (QEMU, `qemu-riscv64`) | `local-sync` | ✅ | `1×16×50257` logits | **44.292 s** real / 42.994 s user / 1.292 s sys |

## How to reproduce

x86:

```bash
time iree-run-module \
  --device=local-task \
  --module=vmfb/gpt2.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

RISC-V / QEMU:

```bash
time iree-run-module-riscv64 \
  --device=local-sync \
  --module=vmfb/gpt2_riscv64.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

## Interpretation

- These are single-call forward-pass timings, not throughput/tokens-per-second numbers
  (there is no decoding loop yet — see the Roadmap).
- The RISC-V number is a **QEMU user-mode emulation timing**, not native RISC-V hardware
  performance. It's useful for validating functional correctness and relative behavior,
  but should not be quoted as a hardware benchmark.
- A fair x86-vs-RISC-V comparison should use identical input, identical `--iree-opt-level`,
  and the same driver (`local-sync` vs `local-task`) on both sides.

## Open items

- [ ] Record x86 baseline timing under the same conditions as the RISC-V run
- [ ] Compare `local-sync` vs `local-task` on both targets
- [ ] Benchmark on real RISC-V hardware (QEMU numbers are not representative of silicon)
- [ ] Compare `-O0` (currently required for RISC-V) vs `-O2` timing/size once the
      LayerNorm lowering issue is resolved upstream
