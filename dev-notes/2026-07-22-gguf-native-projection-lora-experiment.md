---
title: "Experiment: GGUF-Native Projection-to-LoRA Compilation"
date: 2026-07-22
status: proposed
project: representation-engineering
scope: sandbox-prototype
owners:
  - mandersogit
license_intent: Apache-2.0
tags:
  - gguf
  - llama.cpp
  - representation-engineering
  - activation-steering
  - abliteration
  - lora
  - quantization
---

# Experiment: GGUF-Native Projection-to-LoRA Compilation

## Summary

Prototype a tool that accepts a quantized GGUF model and one or more residual-stream direction vectors, then emits a GGUF LoRA adapter implementing a projection-derived weight update without reconstructing the full model in BF16, FP16, or FP32.

The production target is not small models. The intended use is representation engineering on the largest quantized models that can be run locally, including approximately:

- 70B to 130B dense models at Q4-class precision;
- 200B to 300B sparse or mixture-of-experts models at Q4-class precision;
- eventually native FP4/NVFP4 models where their scale representation can be exposed through a compatible weight-source backend.

A small public model will be used only as a correctness and serialization fixture in the constrained sandbox. The implementation must be streaming and memory-bounded from the beginning so that moving to larger hardware changes scale, not architecture.

## Motivation

Existing abliteration and representation-engineering workflows generally assume a Hugging Face Transformers model backed by dense or training-oriented quantization formats. Existing GGUF support commonly means one of the following:

1. dequantize the entire GGUF into a PyTorch model;
2. create an adapter from a BF16/FP16 source checkpoint and later convert it to GGUF;
3. use activation control vectors at inference time rather than compile the intervention into a LoRA;
4. merge or quantize an already-created adapter.

Those approaches do not serve users who can run a model only because it is quantized and who do not have enough RAM or VRAM to materialize the full higher-precision checkpoint. For this project, the quantized GGUF is the authoritative model.

The missing bridge is narrow:

```text
GGUF base weights + residual directions + target tensor selection
    -> bounded streaming contraction
    -> projection-derived LoRA factors
    -> GGUF LoRA adapter
```

## Core hypothesis

For a matrix \(W_q\) reconstructed from the actual quantized GGUF and a unit residual-space direction \(v\), directional projection can be represented exactly as a rank-1 LoRA relative to \(W_q\):

\[
W'_q = (I - \lambda vv^T)W_q
\]

Therefore:

\[
\Delta W_q = W'_q - W_q = -\lambda v(v^T W_q)
\]

Set:

\[
B = -\lambda v
\]

and:

\[
A = v^T W_q
\]

Then:

\[
\Delta W_q = BA
\]

For multiple directions represented by a matrix \(R\), the same construction generalizes to a rank-\(k\) adapter:

\[
B = -\lambda R
\]

\[
A = R^T W_q
\]

The only substantial weight operation required is the left contraction \(R^T W_q\). It can be computed one dequantized row or block range at a time.

## Experiment objective

Demonstrate all of the following in the sandbox:

1. A GGUF tensor can be accessed through a lazy, memory-mapped weight source.
2. The weight source can compute \(R^T W_q\) with bounded peak memory.
3. The resulting LoRA factors are numerically equivalent to explicit projection of the same effective quantized matrix.
4. The factors can be serialized into an adapter accepted by llama.cpp.
5. The implementation does not instantiate a full Transformers model and does not dequantize the complete GGUF.
6. The design is independent of total model size for dense tensors; model size increases I/O and runtime, not peak working memory.

## Sandbox constraints

The initial environment is approximately:

- CPU-only execution;
- about 6 GiB RAM;
- about 39 GiB free disk;
- roughly 5 CPU cores;
- CMake and GCC available;
- Python and CPU PyTorch available;
- no assumption of CUDA, Metal, or large-model inference capability.

These constraints are useful because they force the implementation to avoid accidental full-model materialization.

## Test fixture

Use a small instruction model with two corresponding GGUF artifacts:

- an F16 or BF16 GGUF reference;
- a Q4_K_M or comparable Q4-class GGUF.

A suitable initial candidate is Qwen2.5-0.5B-Instruct, subject to confirming stable source URLs and llama.cpp compatibility at implementation time.

The small model is used only to validate:

- GGUF parsing;
- tensor-name selection;
- quantized deconstruction;
- tensor orientation;
- contraction correctness;
- adapter serialization;
- runtime loading.

No conclusion about refusal behavior or large-model behavioral quality will be drawn from this fixture.

## Proposed software architecture

### `GGUFWeightSource`

Expose the GGUF as a lazy tensor source rather than as a reconstructed model:

```python
class GGUFWeightSource:
    def tensor_info(self, name: str) -> TensorInfo:
        ...

    def iter_dequantized_chunks(
        self,
        name: str,
        *,
        max_chunk_bytes: int,
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[TensorChunk]:
        ...

    def left_contract(
        self,
        name: str,
        directions: torch.Tensor,
        *,
        max_chunk_bytes: int,
    ) -> torch.Tensor:
        """Return directions.T @ W_q without materializing all of W_q."""
```

The implementation should reuse llama.cpp/gguf-py metadata and dequantization logic rather than independently reimplementing GGUF encodings.

### `ProjectionCompiler`

A format-independent component computes adapter factors:

```python
def compile_projection(
    weights: WeightSource,
    tensor_name: str,
    directions: torch.Tensor,
    strength: float,
) -> LoRAFactors:
    a = weights.left_contract(tensor_name, directions)
    b = -strength * directions
    return LoRAFactors(a=a.T, b=b)
```

The compiler should not know whether the base model is GGUF, Safetensors, BitsAndBytes, or a future native NVFP4 representation.

### Adapter output

The initial implementation may emit PEFT-compatible Safetensors and invoke llama.cpp's existing converter to produce a GGUF LoRA. A later implementation may write GGUF LoRA directly if that materially simplifies deployment or preserves canonical tensor naming more reliably.

## Initial tensor targets

For the dense proof of concept, target two ordinary residual-writing matrices:

```text
blk.0.attn_output.weight
blk.0.ffn_down.weight
```

These provide different shapes and exercise the primary matrix orientations used by directional projection.

Subsequent dense-model support should generalize to selected layers and the canonical GGUF tensor classes corresponding to:

- attention output projections;
- FFN down projections;
- optional embeddings or output matrices when required by a particular method.

Packed expert tensors are explicitly deferred until the dense path is proven.

## Direction source for the first experiment

Use a deterministic synthetic unit vector rather than a refusal-derived vector:

```python
generator = torch.Generator().manual_seed(12345)
v = torch.randn(output_dimension, generator=generator, dtype=torch.float32)
v /= torch.linalg.vector_norm(v)
```

This isolates compiler correctness from representation-engineering methodology. The first experiment asks whether a known projection can be compiled and applied correctly, not whether a particular direction modifies refusal behavior.

## Experimental procedure

### Phase 1: repository and dependency setup

1. Add a Python package under `src/`.
2. Pin or record the llama.cpp revision used for gguf-py and adapter conversion.
3. Add reproducible scripts for downloading the test GGUF files.
4. Add `.gitignore` entries for model binaries, llama.cpp checkouts, build output, and generated adapters.
5. Select a permissive project license compatible with the intended work environment.

### Phase 2: synthetic quantized-tensor validation

Before using a real model:

1. Generate small deterministic matrices.
2. Quantize them using supported llama.cpp routines where practical.
3. Dequantize them through the same GGUF/ggml path the production tool will use.
4. Compute \(A = R^T W_q\) both directly and through chunked accumulation.
5. Compare results across multiple chunk boundaries.

This phase should catch:

- shape errors;
- transposition errors;
- row/block alignment errors;
- accumulation precision errors;
- quantization-block boundary mistakes.

### Phase 3: lazy real-GGUF tensor access

1. Memory-map the Q4 test GGUF.
2. Enumerate tensor names, shapes, types, and offsets.
3. Locate the selected attention-output and FFN-down tensors.
4. Dequantize them in bounded chunks.
5. Record peak resident memory.
6. Verify that changing the configured chunk size changes peak temporary memory but not the final result.

Suggested chunk sizes:

- 1 MiB;
- 4 MiB;
- 16 MiB;
- 64 MiB.

### Phase 4: projection equivalence

For deterministic input vectors \(x\), compare:

\[
y_{explicit} = (I - \lambda RR^T)W_qx
\]

with:

\[
y_{LoRA} = W_qx + B(Ax)
\]

Run the comparison for:

- rank 1;
- rank 4;
- F16/BF16 GGUF reference;
- Q4-class GGUF;
- multiple tensor shapes;
- multiple chunk sizes.

Acceptance tolerance should be selected based on the adapter storage dtype and accumulation dtype. FP32 accumulation is required initially.

### Phase 5: adapter serialization

1. Emit the computed factors as PEFT-compatible adapter tensors or directly as GGUF LoRA tensors.
2. Convert to GGUF LoRA using a pinned llama.cpp converter if PEFT is used as the intermediate representation.
3. Inspect the generated adapter metadata and tensor shapes.
4. Confirm canonical base-tensor mappings.

### Phase 6: llama.cpp runtime validation

Build a CPU-only llama.cpp binary and run the small Q4 model:

```bash
llama-cli \
  -m model-Q4_K_M.gguf \
  --lora projection.gguf \
  --seed 12345 \
  --temp 0 \
  -p "Test prompt"
```

Validate that:

- the adapter loads without tensor-name or shape errors;
- inference succeeds;
- enabling the adapter changes deterministic logits or output;
- disabling the adapter restores the baseline;
- a zero-strength adapter reproduces the baseline.

Where practical, expose or compare logits rather than relying only on text divergence.

### Phase 7: quantization-specific comparison

Using the same direction vectors, generate:

\[
A_{F16} = R^T W_{F16}
\]

and:

\[
A_{Q4} = R^T W_{Q4}
\]

Measure:

- cosine similarity;
- relative L2 difference;
- effect on sampled matrix outputs;
- logit divergence when each adapter is applied to the Q4 base;
- whether the exact-GGUF adapter materially outperforms the higher-precision-derived adapter.

This determines whether users need one adapter per exact GGUF artifact or whether an adapter derived from a higher-precision sibling is generally portable across quantizations.

## Memory and scalability requirements

The production algorithm must have peak working memory approximately proportional to:

\[
O(chunk + k d_{in} + k d_{out})
\]

It must not be proportional to total model parameters.

Required invariants:

- the GGUF remains memory-mapped;
- only selected tensors are read;
- only bounded portions of a target tensor are dequantized at once;
- accumulation is performed in FP32 initially;
- the complete dense model is never instantiated;
- the complete target matrix is not required to exist in dense form;
- output size is proportional to adapter rank and target dimensions.

For a 70B to 300B deployment, runtime may scale linearly with bytes read, but peak RAM should remain governed by the configured chunk size and factor dimensions.

## Acceptance criteria

The sandbox experiment succeeds when all of the following are true:

1. **Numerical correctness**
   - Chunked \(R^T W_q\) agrees with whole-tensor reference computation on the small fixture.
   - Base-plus-LoRA agrees with explicit projected weights within documented tolerance.

2. **Bounded memory**
   - Peak RSS remains within the sandbox budget.
   - Reducing chunk size predictably reduces temporary memory.
   - No full-model dense state dictionary is created.

3. **Runtime compatibility**
   - llama.cpp loads the generated GGUF LoRA.
   - The adapter affects inference and can be disabled cleanly.

4. **Reproducibility**
   - A clean checkout can download fixtures, build dependencies, run tests, and reproduce the adapter.
   - Model binaries and generated artifacts are not committed.

5. **Scalable design**
   - The public API is based on a weight-source abstraction.
   - The GGUF implementation streams tensor data.
   - No code path assumes the complete base model fits in memory.

## Non-goals for the sandbox phase

The first experiment will not attempt to solve:

- optimal refusal-direction extraction;
- behavioral refusal evaluation;
- layer and strength optimization;
- full norm-preserving or nonlinear variants requiring large low-rank approximations;
- packed MoE expert adapter execution;
- native NVIDIA NVFP4 tensor-plus-scale ingestion;
- multi-GPU acceleration;
- Metal acceleration;
- support for every GGUF architecture and quantization type;
- an upstream-ready llama.cpp pull request.

These are follow-on phases after the core compiler is validated.

## Follow-on work on larger systems

### AI workstation: 2x RTX 6000 Blackwell

Use for:

- quantized activation collection on 70B to 130B dense models;
- fast refusal-direction experiments;
- logit/KL evaluation;
- adapter strength and layer-range sweeps;
- testing native NVIDIA quantization backends;
- dense models that fit largely or entirely in the aggregate 192 GB of GPU memory.

### Mac Studio M3 Ultra, 512 GB unified memory

Use for:

- very large GGUF inference through Metal;
- 200B to 300B MoE models that benefit from the large unified-memory pool;
- large filesystem-cache-backed scans;
- validation that the compiler remains useful when the model is near the machine's practical inference limit.

### DGX Spark

Use for:

- alternate native quantized inference paths;
- CUDA/Blackwell-specific acceleration experiments;
- cross-platform comparison of direction extraction and evaluation;
- larger automated sweeps where supported.

## MoE follow-on experiment

After dense GGUF LoRA compilation works, investigate packed expert tensors such as:

```text
blk.N.ffn_down_exps.weight
blk.N.ffn_gate_exps.weight
blk.N.ffn_up_exps.weight
```

Questions to resolve:

1. Does current llama.cpp apply LoRA adapters to packed expert operations?
2. Can a shared residual direction be compiled into expert-specific \(A_e\) factors with a shared or repeated \(B\)?
3. What GGUF adapter layout is expected for expert-indexed tensors?
4. Is a runtime change required around `MUL_MAT_ID` or equivalent expert-selection operations?
5. Can expert tensors be streamed expert-by-expert without materializing the packed tensor?

The mathematical operation remains straightforward; runtime adapter support is the principal uncertainty.

## Expected repository layout

```text
representation-engineering/
├── pyproject.toml
├── README.md
├── LICENSE
├── dev-notes/
│   └── 2026-07-22-gguf-native-projection-lora-experiment.md
├── src/
│   └── repreng_gguf/
│       ├── __init__.py
│       ├── reader.py
│       ├── tensor_source.py
│       ├── contraction.py
│       ├── projection.py
│       ├── peft_writer.py
│       └── cli.py
├── tests/
│   ├── test_synthetic_projection.py
│   ├── test_chunked_contraction.py
│   ├── test_real_gguf_tensor.py
│   ├── test_quantization_comparison.py
│   └── test_llama_cpp_adapter.py
├── experiments/
│   └── qwen05b_projection/
│       ├── run.sh
│       ├── prompts.txt
│       └── report.md
└── scripts/
    ├── fetch_llama_cpp.sh
    └── fetch_test_models.sh
```

## Immediate next milestone

Implement a minimal, testable `GGUFWeightSource.left_contract()` and prove the following statement on a real Q4 GGUF tensor:

> A rank-1 GGUF LoRA generated from a streamed contraction of the exact quantized weights produces the same matrix transformation as explicit directional projection, while peak memory remains bounded by the configured chunk size.
