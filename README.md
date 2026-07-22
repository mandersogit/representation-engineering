# representation-engineering

Experimental tooling for compiling residual-space projections into LoRA adapters over quantized GGUF model weights without reconstructing the full model in BF16 or FP32.

The initial milestone validates the projection-to-LoRA mathematics and bounded row-chunk accumulation. GGUF parsing and adapter serialization are added as separate, tested milestones.
