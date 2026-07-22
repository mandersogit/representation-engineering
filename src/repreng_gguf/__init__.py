"""GGUF-native representation-engineering primitives."""

from .projection import (
    LoRAFactors,
    apply_lora,
    compile_projection_factors,
    explicit_project_weight,
    left_contract_chunked,
    orthonormalize_directions,
)

__all__ = [
    "LoRAFactors",
    "apply_lora",
    "compile_projection_factors",
    "explicit_project_weight",
    "left_contract_chunked",
    "orthonormalize_directions",
]
