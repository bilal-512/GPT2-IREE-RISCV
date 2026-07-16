#!/usr/bin/env bash
#
# Compile the imported GPT-2 MLIR into VMFB executables for both x86 and RISC-V,
# then run a smoke test on x86.
#
# Prerequisites:
#   - model/gpt2.onnx has been produced by scripts/export_gpt2.py
#   - IREE tools (iree-import-onnx, iree-compile, iree-run-module) are on PATH
#
# Usage:
#   bash scripts/compile.sh

set -euo pipefail

mkdir -p mlir vmfb logs

echo "==> Importing ONNX -> MLIR"
iree-import-onnx \
  model/gpt2.onnx \
  -o mlir/gpt2.mlir

grep -n "func.func @main_graph" mlir/gpt2.mlir || true

echo "==> Compiling for x86 (host)"
iree-compile \
  mlir/gpt2.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  -o vmfb/gpt2.vmfb

echo "==> Running smoke test on x86"
iree-run-module \
  --device=local-task \
  --module=vmfb/gpt2.vmfb \
  --function=main_graph \
  --input="1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  > logs/gpt2_x86.log 2>&1
echo "x86 run complete, see logs/gpt2_x86.log"

echo "==> Cross-compiling for RISC-V (RV64GC)"
# NOTE: default optimization fails while lowering onnx.LayerNormalization on this
# backend. --iree-opt-level=O0 is required until that is fixed upstream.
# See docs/troubleshooting.md.
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

echo "==> Done."
echo "    x86 artifact:    vmfb/gpt2.vmfb"
echo "    RISC-V artifact: vmfb/gpt2_riscv64.vmfb"
echo ""
echo "To run on RISC-V under QEMU:"
echo "  time iree-run-module-riscv64 \\"
echo "    --device=local-sync \\"
echo "    --module=vmfb/gpt2_riscv64.vmfb \\"
echo "    --function=main_graph \\"
echo "    --input=\"1x16xi64=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\""
