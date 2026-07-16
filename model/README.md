# model/

Generated artifact goes here: `gpt2.onnx`

Not committed to the repository (see `.gitignore`) since it can be regenerated with:

```bash
python scripts/export_gpt2.py
```

Expected file: `model/gpt2.onnx` — input `input_ids` (`int64`, dynamic sequence length),
output `logits` (opset 17).
