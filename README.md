# representation-engineering

Experimental tooling for compiling residual-space projections into native llama.cpp GGUF LoRA adapters over quantized GGUF model weights, without reconstructing the full model in BF16, FP16, or FP32.

The intended workload is the largest quantized model a local system can use: 70B–130B dense models and, after packed-expert support is verified, 200B–300B MoE models.

## Current status

The prototype currently provides:

- mmap-backed GGUF v2/v3 tensor-directory parsing;
- bounded row-chunk dequantization and `R.T @ W_q` contraction;
- F32, F16, BF16, Q4_0, Q8_0, Q4_K, and Q6_K decoding;
- rank-1 and rank-k projection-to-LoRA compilation;
- direct native GGUF LoRA serialization, without a PEFT intermediate;
- structural validation mirroring llama.cpp's adapter metadata, pairing, and shape checks;
- a CLI for inspection, compilation, and validation;
- numerical and end-to-end artifact tests.

The emitted adapter layout is based on llama.cpp revision `1a064ab0921238c1daa397d6f4a900ef33884de2`:

- `general.type = adapter`
- `general.architecture` matches the base GGUF
- `adapter.type = lora`
- `adapter.lora.alpha = rank`
- tensors are named `<base tensor>.lora_a` and `<base tensor>.lora_b`

The sandbox used for the initial experiment cannot download or build llama.cpp or a public model GGUF through its normal network path. The format has therefore been checked against llama.cpp source and independent GGUF fixtures, but runtime loading by `llama-cli` remains a required validation on a less constrained host.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Inspect a model

List projection-compatible rank-2 tensors:

```bash
repreng-gguf inspect model-Q4_K_M.gguf --supported-only
```

Filter to likely residual-writing targets:

```bash
repreng-gguf inspect model-Q4_K_M.gguf \
  --pattern 'attn_output|ffn_down' \
  --supported-only
```

## Direction archive contract

Directions are supplied as a Safetensors file. Each key must be the exact canonical tensor name in the base GGUF. Each value must have shape:

- `(out_features,)` for rank 1; or
- `(out_features, rank)` for rank k.

Example:

```python
from safetensors.torch import save_file
import torch

save_file(
    {
        "blk.12.attn_output.weight": torch.randn(5120, 4),
        "blk.12.ffn_down.weight": torch.randn(5120, 4),
    },
    "directions.safetensors",
)
```

The compiler orthonormalizes the columns before calculating:

```text
A = R.T @ W_q
B = -strength * R
```

The resulting adapter implements `W_q + B @ A`.

## Compile a native GGUF LoRA

```bash
repreng-gguf compile \
  model-Q4_K_M.gguf \
  directions.safetensors \
  projection-lora.gguf \
  --strength 1.0 \
  --chunk-mib 64 \
  --output-type f16
```

`--chunk-mib` controls the approximate raw-plus-dequantized row-chunk budget. The model remains memory-mapped and only selected tensor rows are materialized.

## Validate the adapter

```bash
repreng-gguf validate \
  model-Q4_K_M.gguf \
  projection-lora.gguf
```

JSON output is available with `--json`.

The validator checks the conditions enforced by llama.cpp's adapter loader:

- adapter metadata and architecture match;
- every `.lora_a` has a `.lora_b`;
- every target exists in the base model;
- A/B dimensions match the base matrix;
- rank and alpha scaling are internally consistent.

## Memory benchmark

The benchmark creates a sparse but structurally valid Q4_K GGUF and contracts it in isolated worker processes:

```bash
PYTHONPATH=src python experiments/benchmark_sparse_q4k.py \
  --rows 32768 \
  --width 4096 \
  --rank 4 \
  --chunk-mib 1 8 64
```

In the initial sandbox run, the tensor occupied 72 MiB in Q4_K and represented 512 MiB of dense FP32 weights. Peak incremental RSS was approximately:

| Chunk budget | Peak incremental RSS | Elapsed |
|---:|---:|---:|
| 1 MiB | 83 MiB | 0.71 s |
| 8 MiB | 112 MiB | 0.54 s |
| 64 MiB | 311 MiB | 0.79 s |

The benchmark uses zero-filled sparse tensor data to isolate container parsing, mmap access, decoding, chunking, and contraction memory behavior. It is not a model-quality benchmark.

## Tests

The current test suite covers:

- projection equivalence for rank 1 and rank k;
- chunk-boundary invariance;
- manual Q4_0, Q8_0, Q4_K, and Q6_K block decoding;
- mmap-backed GGUF row views;
- streamed contractions over mixed tensor types;
- direct GGUF LoRA serialization;
- CLI compile/inspect/validate workflow;
- adapter architecture and shape validation.

## Known limitations

- No actual refusal-direction extraction is implemented yet.
- Runtime loading by `llama-cli` remains to be performed on an external host.
- Packed MoE expert tensors and `MUL_MAT_ID` LoRA execution are not yet validated.
- Q5_K and less common IQ/TQ formats are not yet implemented.
- Native NVFP4/MXFP4 scale representations require a separate weight-source backend or additional GGUF decoding work.
- The current parser handles a single GGUF file; split/sharded GGUF sets are not yet joined.

See `dev-notes/2026-07-22-gguf-native-projection-lora-experiment.md` for the experiment plan.
