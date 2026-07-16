"""
Sanity-check the exported GPT-2 ONNX model with ONNX Runtime.

Usage:
    python test_onnx.py

Expected output:
    Output shape: (1, 16, 50257)
"""

import numpy as np
import onnxruntime as ort

MODEL_PATH = "model/gpt2.onnx"


def main():
    sess = ort.InferenceSession(MODEL_PATH)

    x = np.random.randint(0, 1000, (1, 16), dtype=np.int64)

    y = sess.run(None, {"input_ids": x})

    print("Output shape:", y[0].shape)


if __name__ == "__main__":
    main()
