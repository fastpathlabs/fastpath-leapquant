#!/usr/bin/env python3
"""
record_trace.py

Records what a Qwen3.5 model feeds to its Gated DeltaNet kernel while it
generates text, one file per layer. The files are replayed on a CPU by
leapquant_replay (see leapquant_replay.cpp for the file format).

Why
    The replay shows how far LeapQuant and the simpler storage formats drift
    from full precision on the states of a real model, before any GPU kernel
    is written.

Two ways to run it

  1. Self-test. Needs only numpy. Writes a small synthetic trace whose
     "model output" is computed here with plain numpy. Use it to check that
     this script and leapquant_replay agree on the file format and the math.

         python3 record_trace.py --self-test selftest.bin
         ./leapquant_replay selftest.bin

  2. Real model. Needs a GPU, torch and transformers.

         python3 record_trace.py --model Qwen/Qwen3.5-9B \
             --prompt "Explain how a B-tree splits a node, step by step." \
             --max-new-tokens 4096 --layers 0,13,30 --out-dir traces

     Layer numbers are positions in the model. Only linear-attention layers
     can be recorded; the script lists them when it starts.

How the real-model recording works
    transformers runs single-token decoding of a Gated DeltaNet layer through
    the function torch_recurrent_gated_delta_rule() in
    transformers/models/qwen3_5/modeling_qwen3_5.py. This script replaces that
    function with a wrapper that calls the original and writes down
        the state before the first decode token   (once per layer)
        decay, beta, key, value, query            (every token, every head)
        the kernel's own output                   (every token, every head)
    key and query are written after the same normalisation and scaling the
    kernel applies internally, so the trace holds exactly the numbers that
    enter the recurrence.

    The trace carries the kernel's own output for every token, and
    leapquant_replay prints a self-check line that compares its fp32 replay
    with that output. If the self-check says "ok", the trace is right.

    STATUS: written against transformers 5.22.2. Run so far only on
    Qwen/Qwen3.5-0.8B on a CPU (--device cpu), where the self-check passes.
    Not yet run on Qwen3.5-9B or on a GPU.

    This script assumes transformers runs its own PyTorch kernel. If the
    flash-linear-attention package is installed, transformers uses that
    package's kernel instead; do not install it in the recording environment.
"""

import argparse
import os
import struct
import sys

import numpy as np

MAGIC = b"LQTRACE1"


class TraceWriter:
    """Writes one trace file. Arrays are converted to little-endian float32."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "wb")
        self.tokens_written = 0
        self.header_written = False

    def write_header(self, key_channels, value_channels, head_count, decay_values_per_head,
                     has_initial_state, has_expected_output):
        self.file.write(MAGIC)
        self.file.write(struct.pack("<6i", key_channels, value_channels, head_count,
                                    decay_values_per_head, int(has_initial_state), int(has_expected_output)))
        self.header_written = True

    def write_initial_state(self, state):
        """state: [heads, key_channels, value_channels]"""
        np.ascontiguousarray(state, dtype="<f4").tofile(self.file)

    def write_token(self, decay, beta, key, value, query, expected_output):
        """
        decay            [heads, 1] or [heads, key_channels]
        beta             [heads]
        key, query       [heads, key_channels]
        value            [heads, value_channels]
        expected_output  [heads, value_channels]
        """
        beta_column = np.asarray(beta, dtype="<f4").reshape(-1, 1)
        block = np.concatenate(
            [np.asarray(decay, dtype="<f4"), beta_column, np.asarray(key, dtype="<f4"),
             np.asarray(value, dtype="<f4"), np.asarray(query, dtype="<f4"),
             np.asarray(expected_output, dtype="<f4")],
            axis=1,
        )
        np.ascontiguousarray(block, dtype="<f4").tofile(self.file)
        self.tokens_written += 1

    def close(self):
        self.file.close()


def run_self_test(path, heads=4, key_channels=32, value_channels=48, tokens=400, seed=0):
    """Writes a synthetic Gated DeltaNet trace computed with plain numpy."""
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((heads, key_channels, value_channels)).astype(np.float32)

    writer = TraceWriter(path)
    writer.write_header(key_channels, value_channels, heads, 1, True, True)
    writer.write_initial_state(state)

    for _ in range(tokens):
        alpha = (1.0 - 10.0 ** rng.uniform(-3.0, -1.3, size=heads)).astype(np.float32)
        beta = rng.uniform(0.2, 1.0, size=heads).astype(np.float32)
        key = rng.standard_normal((heads, key_channels)).astype(np.float32)
        key /= np.linalg.norm(key, axis=1, keepdims=True)
        query = rng.standard_normal((heads, key_channels)).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        query /= np.float32(np.sqrt(key_channels))
        value = rng.standard_normal((heads, value_channels)).astype(np.float32)

        # The same five lines as torch_recurrent_gated_delta_rule() in transformers.
        state = state * alpha[:, None, None]
        held = (state * key[:, :, None]).sum(axis=1)
        delta = (value - held) * beta[:, None]
        state = state + key[:, :, None] * delta[:, None, :]
        output = (state * query[:, :, None]).sum(axis=1)

        writer.write_token(alpha[:, None], beta, key, value, query, output)

    writer.close()
    print(f"wrote {path}: {tokens} tokens, {heads} heads, state {key_channels} x {value_channels}")


def record_from_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    # Find the linear-attention layers and remember which one is running.
    current_layer = {"index": None}
    linear_attention_layers = []

    def remember_layer(module, _inputs):
        current_layer["index"] = module.layer_idx

    for module in model.modules():
        if module.__class__.__name__ == "Qwen3_5GatedDeltaNet":
            module.register_forward_pre_hook(remember_layer)
            linear_attention_layers.append(module.layer_idx)

    if not linear_attention_layers:
        sys.exit("error: this model has no Qwen3_5GatedDeltaNet layers")
    print(f"linear-attention layers in this model: {linear_attention_layers}")

    if args.layers == "all":
        wanted_layers = set(linear_attention_layers)
    else:
        wanted_layers = set(int(text) for text in args.layers.split(","))
        not_linear = sorted(wanted_layers - set(linear_attention_layers))
        if not_linear:
            sys.exit(f"error: layers {not_linear} are not linear-attention layers")

    writers = {}
    original_kernel = qwen_module.torch_recurrent_gated_delta_rule

    def recording_kernel(query, key, value, g, beta, initial_state=None, output_final_state=False,
                         use_qk_l2norm_in_kernel=False, **kwargs):
        layer = current_layer["index"]
        single_token_decode = query.shape[0] == 1 and query.shape[1] == 1 and initial_state is not None
        if layer not in wanted_layers or not single_token_decode:
            return original_kernel(
                query, key, value, g=g, beta=beta, initial_state=initial_state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

        # The kernel computes in float32 and only at the very end rounds its
        # output to the dtype of `query` (bfloat16 for a model loaded in
        # bfloat16). The trace needs the output BEFORE that rounding: bfloat16
        # keeps 8 bits, so the rounded output is 0.4% off and the self-check of
        # leapquant_replay would fail on a correct trace. So the kernel is
        # called with float32 inputs, which makes it return float32, and the
        # rounding is done here. The model receives exactly the same numbers.
        model_dtype = query.dtype
        output_float32, final_state = original_kernel(
            query.float(), key.float(), value.float(), g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        output = output_float32.to(model_dtype)

        with torch.no_grad():
            # Shapes here: query/key [1, 1, heads, key_channels], value [1, 1, heads, value_channels],
            # g/beta [1, 1, heads], initial_state [1, heads, key_channels, value_channels].
            query_used = query[0, 0].float()
            key_used = key[0, 0].float()
            if use_qk_l2norm_in_kernel:
                query_used = query_used * torch.rsqrt((query_used * query_used).sum(dim=-1, keepdim=True) + 1e-6)
                key_used = key_used * torch.rsqrt((key_used * key_used).sum(dim=-1, keepdim=True) + 1e-6)
            query_used = query_used / (query_used.shape[-1] ** 0.5)
            # Some transformers versions hand the kernel one key/query per KEY head and let
            # it share them between value heads. The trace holds one per value head.
            value_heads = value.shape[-2]
            if key_used.shape[0] != value_heads:
                repeats = value_heads // key_used.shape[0]
                key_used = key_used.repeat_interleave(repeats, dim=0)
                query_used = query_used.repeat_interleave(repeats, dim=0)
            alpha = g[0, 0].float().exp()

            if layer not in writers:
                heads, key_channels = key_used.shape
                value_channels = value.shape[-1]
                writer = TraceWriter(os.path.join(args.out_dir, f"trace_layer_{layer:02d}.bin"))
                writer.write_header(key_channels, value_channels, heads, 1, True, True)
                writer.write_initial_state(initial_state[0].float().cpu().numpy())
                writers[layer] = writer

            writers[layer].write_token(
                alpha.reshape(-1, 1).cpu().numpy(),
                beta[0, 0].float().cpu().numpy(),
                key_used.cpu().numpy(),
                value[0, 0].float().cpu().numpy(),
                query_used.cpu().numpy(),
                output_float32[0, 0].float().cpu().numpy(),
            )
        return output, final_state

    qwen_module.torch_recurrent_gated_delta_rule = recording_kernel
    try:
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as handle:
                prompt = handle.read()
        else:
            prompt = args.prompt
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        print(f"prompt: {inputs['input_ids'].shape[1]} tokens. generating {args.max_new_tokens} tokens ...")
        with torch.no_grad():
            # min_new_tokens = max_new_tokens keeps the model generating past its
            # end-of-text token, so the trace has the requested length.
            model.generate(**inputs, max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens,
                           do_sample=False)
    finally:
        qwen_module.torch_recurrent_gated_delta_rule = original_kernel

    if not writers:
        sys.exit("error: the kernel wrapper was never called for a single-token decode step.\n"
                 "       transformers may have changed how the layer calls its kernel, or a compiled\n"
                 "       kernel from the Hub is in use. Check Qwen3_5GatedDeltaNet.forward().")
    for layer in sorted(writers):
        writer = writers[layer]
        writer.close()
        size_mb = os.path.getsize(writer.path) / 1e6
        print(f"wrote {writer.path}: {writer.tokens_written} tokens, {size_mb:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="Record Gated DeltaNet kernel inputs of a Qwen3.5 model.")
    parser.add_argument("--self-test", metavar="PATH", help="write a synthetic trace (numpy only) and exit")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="Write a detailed, step-by-step explanation of how a B-tree splits a node.")
    parser.add_argument("--prompt-file", help="read the prompt from this file instead of --prompt")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--layers", default="all", help='comma-separated layer numbers, or "all"')
    parser.add_argument("--out-dir", default="traces")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.self_test)
        return
    record_from_model(args)


if __name__ == "__main__":
    main()
