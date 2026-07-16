# vmfb/

Compiled IREE VM FlatBuffer artifacts go here:

- `gpt2.vmfb` — compiled for x86 (`--iree-llvmcpu-target-cpu=host`)
- `gpt2_riscv64.vmfb` — cross-compiled for RISC-V RV64GC (`embedded-elf-riscv_64`,
  compiled with `--iree-opt-level=O0`, see `docs/troubleshooting.md`)

Not committed to the repository (see `.gitignore`) — regenerate with:

```bash
bash scripts/compile.sh
```

Do not interchange the x86 and RISC-V artifacts; each embeds target-specific code and
will fail (`Exec format error` / wrong HAL target) if run on the other platform.
