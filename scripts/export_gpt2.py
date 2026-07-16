"""
Export GPT-2 Small (124M) from PyTorch to ONNX.

Usage:
    python scripts/export_gpt2.py

Output:
    model/gpt2.onnx
"""

import torch
from transformers import GPT2LMHeadModel

OUTPUT_PATH = "model/gpt2.onnx"


def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    dummy = torch.randint(0, 1000, (1, 16), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy,
        OUTPUT_PATH,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "logits": {1: "sequence"},
        },
    )

    print("Export complete")


if __name__ == "__main__":
    main()
