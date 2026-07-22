---
title: "Sandbox Prototype Results: GGUF-Native Projection LoRA"
date: 2026-07-22
status: milestone-complete
project: representation-engineering
scope: sandbox-prototype
base_llama_cpp_revision: 1a064ab0921238c1daa397d6f4a900ef33884de2
tags:
  - gguf
  - llama.cpp
  - lora
  - projection
  - quantization
  - benchmark
---

# Sandbox Prototype Results: GGUF-Native Projection LoRA

## Outcome

The initial sandbox experiment succeeded at its primary engineering objective:

> A residual-space projection can be compiled directly from the effective weights in a quantized GGUF into a native llama.cpp GGUF LoRA without loading a Transformers model, obtaining the BF16 checkpoint, or materializing the complete target matrix.

The prototype now implements the complete format-level path:

```text
quantized GGUF base
  -> mmap selected tensor rows
  -> bounded dequantization
  -> accumulate R.T @ W_q
  -> construct A and B LoRA factors
  -> write native GGUF LoRA
  -> structurally validate against the base GGUF
```

The remaining major validation is to load the generated adapter with an actual `llama-cli` build and evaluate it on a real model. That cannot be completed in the current sandbox because its normal runtime cannot resolve or download from GitHub or Hugging Face.

## Implemented components

### Projection mathematics

For an orthonormal direction matrix `R` and a quantized effective matrix `W_q`, the compiler calculates:

```text
A = R.T @ W_q
B = -strength * R
```

so that:

```text
W_q + B @ A = W_q - strength * R @ R.T @ W_q
```

Rank-1 and rank-k cases are supported. Direction columns are orthonormalized with QR before compilation.

### Memory-bounded GGUF access

The GGUF base remains memory-mapped. The implementation reads complete logical rows in bounded groups, dequantizes only those rows, accumulates into the small `A` matrix, and discards the temporary chunk.

Peak working memory is governed principally by:

```text
row chunk + dequantized chunk + direction matrix + output factors
```

rather than total model size.

### Quantization support

The prototype currently decodes:

- F32
- F16
- BF16
- Q4_0
- Q8_0
- Q4_K
- Q6_K

Q4_K and Q6_K are the central formats needed for ordinary mixed Q4_K_M files. Their block layouts were implemented from the pinned llama.cpp MIT source and tested with independently packed blocks that exercise subblock scales, minima, low bits, and high bits.

### Native GGUF LoRA output

The adapter writer emits the format expected by llama.cpp:

```text
general.type = adapter
general.architecture = <base architecture>
adapter.type = lora
adapter.lora.alpha = rank
```

Tensor pairs are named:

```text
<base tensor>.lora_a
<base tensor>.lora_b
```

The factor shapes are:

```text
A: (rank, in_features)
B: (out_features, rank)
```

The intervention strength is already included in `B`. Setting alpha equal to rank makes llama.cpp's runtime `alpha / rank` multiplier equal to one.

### Structural validator

The validator mirrors the relevant checks in `src/llama-adapter.cpp`:

- adapter metadata is present and correct;
- adapter architecture matches the base model;
- every A tensor has a B tensor;
- every target exists in the base GGUF;
- base, A, and B dimensions are compatible;
- ranks are consistent;
- alpha/rank scaling is reported.

### CLI

The current command surface is:

```bash
repreng-gguf inspect BASE.gguf --supported-only
repreng-gguf compile BASE.gguf DIRECTIONS.safetensors ADAPTER.gguf
repreng-gguf validate BASE.gguf ADAPTER.gguf
```

The Safetensors direction archive uses exact GGUF base tensor names as keys. Values have shape `(out_features,)` or `(out_features, rank)`.

## Test results

The complete local suite currently reports:

```text
52 passed
```

Coverage includes:

- rank-1 and rank-k projection equivalence;
- chunk-boundary invariance;
- explicit versus LoRA-applied matrix equality;
- manual Q4_0 and Q8_0 decoding;
- manual Q4_K scale/minimum and nibble decoding;
- manual Q6_K low/high-bit and signed-scale decoding;
- GGUF metadata, shape, alignment, and tensor-offset parsing;
- verification that raw row chunks remain mmap-backed views;
- streamed contraction for F16, Q4_0, Q8_0, Q4_K, and Q6_K;
- direct GGUF LoRA serialization and F16 round trip;
- CLI inspect/compile/validate operation;
- validator detection of architecture mismatch.

## Memory benchmark

A sparse but structurally valid Q4_K GGUF was created with one matrix of shape:

```text
32768 x 4096
```

Its sizes were:

```text
Q4_K storage: 72 MiB
Dense FP32 equivalent: 512 MiB
Direction rank: 4
```

Measured results in isolated worker processes:

| Chunk budget | Peak RSS before contraction | Peak RSS | Increment | Elapsed |
|---:|---:|---:|---:|---:|
| 1 MiB | 291.6 MiB | 374.9 MiB | 83.3 MiB | 0.71 s |
| 8 MiB | 291.4 MiB | 403.8 MiB | 112.3 MiB | 0.54 s |
| 64 MiB | 291.5 MiB | 602.6 MiB | 311.1 MiB | 0.79 s |

The test confirms:

1. the operation does not require the 512 MiB dense matrix to remain resident;
2. peak memory responds to the configured chunk size;
3. mmap and quantized row streaming are functioning;
4. the output remains only rank by input width.

The zero-filled sparse fixture isolates format and memory behavior. It does not estimate refusal quality or realistic storage throughput for non-sparse model files.

## Sandbox limitations encountered

The sandbox runtime could not resolve or directly download from:

- `github.com`
- `huggingface.co`

The GitHub connector remained available for source inspection and incremental commits. Web browsing could inspect public model repositories, but did not provide a binary file path usable by the Python/container runtime.

Consequently, the following were not performed here:

- cloning and building llama.cpp;
- downloading a public model GGUF;
- loading the generated adapter in `llama-cli`;
- comparing model logits with and without the adapter;
- deriving real refusal directions from model activations.

These are environment limitations, not known failures of the implementation.

## Confidence assessment

Current confidence is high for:

- the projection-to-LoRA mathematics;
- row orientation and dimensions;
- bounded contraction behavior;
- supported quantization decoding on tested block layouts;
- GGUF container parsing and writing;
- structural agreement with the pinned llama.cpp adapter loader.

Current confidence is medium until external-host validation for:

- actual llama.cpp runtime loading;
- canonical target selection on a real 70B+ model;
- behavior of adapters over Q4_K_M mixed tensors;
- performance when scanning hundreds of gigabytes from real storage.

Confidence remains low or unresolved for:

- packed MoE expert LoRA execution;
- split GGUF sets;
- native NVFP4/MXFP4 representations;
- behavioral efficacy of any particular refusal-direction method.

## Next experiment on external hardware

The next useful experiment should use a real dense GGUF, preferably 70B or larger, on the AI workstation or Mac Studio:

1. clone this repository and llama.cpp;
2. install the package and run the test suite;
3. inspect the model's `attn_output` and `ffn_down` tensors;
4. create deterministic synthetic directions for one layer;
5. compile a rank-1 or rank-4 adapter;
6. validate it structurally;
7. load it with `llama-cli --lora`;
8. compare deterministic logits or token probabilities;
9. verify memory and I/O telemetry during compilation;
10. only after runtime equivalence is established, connect real refusal-direction extraction.

For MoE models, first determine whether current llama.cpp applies LoRA factors to packed expert tensors used through `MUL_MAT_ID`. If not, dense/shared residual writers can still be tested while the runtime support question is isolated.

## Key published milestones

Representative commits on `main` include:

- projection core: `f595c6d48ab856e08a951ebfd9a0689df5060c15`
- mmap GGUF reader: `79e5a38f96b4eb5c2c65f61244e73e2eb6087f76`
- streamed weight source: `f1f91dc3ecae64545c6c8f79ed9a27d488239784`
- Q4_K support: `45a8967f063a1ce2696896328bfbfba837a0d0c7`
- Q6_K support: `558ebab4f1da559266930d54098dbf6531ecb1bd`
- direct GGUF LoRA writer: `e308d2f27ca61cef0fef18bd91714e9a11f8544a`
- adapter serialization tests: `ce47875cf15cd5c710aea342b0c72be07cb0dbfa`
- CLI: `4e7e790f934394b9bdcb0b5c3b4fae5cb0a76e39`
- validator: `c7cff4c42768856d90d2f6fb9f190503dcb4b2ee`
- memory benchmark: `4016f3830090d3f96308f73562cea30d34005ec2`
