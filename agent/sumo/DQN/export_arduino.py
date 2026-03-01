"""
Export trained Q-Model weights as a C header file for Arduino Nano.

Usage:
    python export_arduino.py [--model-dir tmp/q_model] [--output q_model_weights.h]

Produces a header with:
  - Weight/bias arrays stored in PROGMEM (flash)
  - A simple inference function (forward pass with ReLU)
  - Memory usage report
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

import argparse
import numpy as np
import torch
from q_model import DQNAgent, QModelConfig


def export_c_header(agent: DQNAgent, output_path: str):
    """Export Q-network weights to a C header file for Arduino."""

    sd = agent.q_net.state_dict()

    # Extract weight matrices and bias vectors
    layers = [
        ("W1", sd["net.0.weight"].cpu().numpy(), sd["net.0.bias"].cpu().numpy()),  # Linear(5,16)
        ("W2", sd["net.2.weight"].cpu().numpy(), sd["net.2.bias"].cpu().numpy()),  # Linear(16,16)
        ("W3", sd["net.4.weight"].cpu().numpy(), sd["net.4.bias"].cpu().numpy()),  # Linear(16,5)
    ]

    total_params = 0
    total_bytes = 0

    lines = []
    lines.append("// Auto-generated Q-Model weights for Arduino Nano")
    lines.append("// Network: 5 -> 16 -> 16 -> 5  (ReLU activations)")
    lines.append("// Actions: 0=backward, 1=forward, 2=stop, 3=spin_left, 4=spin_right")
    lines.append(f"// Total parameters: {agent.param_count()}")
    lines.append("")
    lines.append("#ifndef Q_MODEL_WEIGHTS_H")
    lines.append("#define Q_MODEL_WEIGHTS_H")
    lines.append("")
    lines.append("#include <avr/pgmspace.h>")
    lines.append("")

    # Dimensions
    lines.append("#define Q_INPUT_DIM   5")
    lines.append("#define Q_HIDDEN1    16")
    lines.append("#define Q_HIDDEN2    16")
    lines.append("#define Q_OUTPUT_DIM  5")
    lines.append("")

    # Action map
    lines.append("// Action -> motor values [left, right]")
    lines.append("const float ACTION_MAP[Q_OUTPUT_DIM][2] PROGMEM = {")
    lines.append("    {-1.0f, -1.0f},  // 0: backward")
    lines.append("    { 1.0f,  1.0f},  // 1: forward")
    lines.append("    { 0.0f,  0.0f},  // 2: stop")
    lines.append("    {-1.0f,  1.0f},  // 3: spin left")
    lines.append("    { 1.0f, -1.0f},  // 4: spin right")
    lines.append("};")
    lines.append("")

    # Write each layer
    for name, weight, bias in layers:
        rows, cols = weight.shape
        n_w = weight.size
        n_b = bias.size
        total_params += n_w + n_b
        total_bytes += (n_w + n_b) * 4

        # Weight matrix (stored row-major: [out_features][in_features])
        lines.append(f"// {name} weight: [{rows}][{cols}]  ({n_w} params)")
        lines.append(f"const float {name}_weight[{rows}][{cols}] PROGMEM = {{")
        for r in range(rows):
            vals = ", ".join(f"{v: .6f}f" for v in weight[r])
            comma = "," if r < rows - 1 else ""
            lines.append(f"    {{{vals}}}{comma}")
        lines.append("};")
        lines.append("")

        # Bias vector
        lines.append(f"// {name} bias: [{n_b}]")
        vals = ", ".join(f"{v: .6f}f" for v in bias)
        lines.append(f"const float {name}_bias[{n_b}] PROGMEM = {{{vals}}};")
        lines.append("")

    # Inference function
    lines.append("// -- Inference -------------------------------------------------")
    lines.append("// Forward pass: returns best action index (0-4)")
    lines.append("// input[5] = {front, f_right, f_left, last_left_motor, last_right_motor}")
    lines.append("")
    lines.append("static inline void matmul_relu(")
    lines.append("    const float* input, int in_dim,")
    lines.append("    const float weight[][16], const float* bias, int out_dim,  // max hidden=16")
    lines.append("    float* output)")
    lines.append("{")
    lines.append("    for (int o = 0; o < out_dim; o++) {")
    lines.append("        float sum = pgm_read_float(&bias[o]);")
    lines.append("        for (int i = 0; i < in_dim; i++) {")
    lines.append("            sum += input[i] * pgm_read_float(&weight[o][i]);")
    lines.append("        }")
    lines.append("        output[o] = (sum > 0.0f) ? sum : 0.0f;  // ReLU")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("int q_model_predict(const float input[Q_INPUT_DIM]) {")
    lines.append("    float h1[Q_HIDDEN1];")
    lines.append("    float h2[Q_HIDDEN2];")
    lines.append("    float qvals[Q_OUTPUT_DIM];")
    lines.append("")
    lines.append("    // Layer 1: input(5) -> h1(16) + ReLU")
    lines.append("    for (int o = 0; o < Q_HIDDEN1; o++) {")
    lines.append("        float sum = pgm_read_float(&W1_bias[o]);")
    lines.append("        for (int i = 0; i < Q_INPUT_DIM; i++) {")
    lines.append("            sum += input[i] * pgm_read_float(&W1_weight[o][i]);")
    lines.append("        }")
    lines.append("        h1[o] = (sum > 0.0f) ? sum : 0.0f;")
    lines.append("    }")
    lines.append("")
    lines.append("    // Layer 2: h1(16) -> h2(16) + ReLU")
    lines.append("    for (int o = 0; o < Q_HIDDEN2; o++) {")
    lines.append("        float sum = pgm_read_float(&W2_bias[o]);")
    lines.append("        for (int i = 0; i < Q_HIDDEN1; i++) {")
    lines.append("            sum += h1[i] * pgm_read_float(&W2_weight[o][i]);")
    lines.append("        }")
    lines.append("        h2[o] = (sum > 0.0f) ? sum : 0.0f;")
    lines.append("    }")
    lines.append("")
    lines.append("    // Layer 3: h2(16) -> qvals(5) (no activation)")
    lines.append("    for (int o = 0; o < Q_OUTPUT_DIM; o++) {")
    lines.append("        float sum = pgm_read_float(&W3_bias[o]);")
    lines.append("        for (int i = 0; i < Q_HIDDEN2; i++) {")
    lines.append("            sum += h2[i] * pgm_read_float(&W3_weight[o][i]);")
    lines.append("        }")
    lines.append("        qvals[o] = sum;")
    lines.append("    }")
    lines.append("")
    lines.append("    // Argmax")
    lines.append("    int best = 0;")
    lines.append("    for (int i = 1; i < Q_OUTPUT_DIM; i++) {")
    lines.append("        if (qvals[i] > qvals[best]) best = i;")
    lines.append("    }")
    lines.append("    return best;")
    lines.append("}")
    lines.append("")
    lines.append("#endif // Q_MODEL_WEIGHTS_H")
    lines.append("")

    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"Arduino C header exported to: {output_path}")
    print(f"{'='*60}")
    print(f"  Total parameters : {total_params}")
    print(f"  Flash usage      : {total_bytes} bytes ({total_bytes/1024:.1f} KB) for weights")
    print(f"  SRAM at inference: ~{16*4 + 16*4 + 5*4 + 5*4} bytes (hidden + output buffers)")
    print(f"  Arduino Nano fit : {'YES' if total_bytes < 30000 else 'TIGHT'}")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(description="Export Q-Model to Arduino C header")
    ap.add_argument("--model-dir", type=str, default="tmp/q_model")
    ap.add_argument("--output", type=str, default="tmp/q_model/q_model_weights.h")
    args = ap.parse_args()

    cfg = QModelConfig(
        chkpt_dir=args.model_dir,
        obs_dim=5,
        num_actions=5,
        hidden1=16,
        hidden2=16,
    )

    agent = DQNAgent(cfg)
    agent.load_models()

    export_c_header(agent, args.output)


if __name__ == "__main__":
    main()
