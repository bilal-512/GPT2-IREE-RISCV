# mlir/

Imported MLIR goes here: `gpt2.mlir` (produced by `iree-import-onnx`, exposing
`func.func @main_graph`). Not committed by default — regenerate with:

```bash
iree-import-onnx model/gpt2.onnx -o mlir/gpt2.mlir
```
