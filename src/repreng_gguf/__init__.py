"""GGUF-native representation-engineering primitives."""

from .gguf_reader import GGUFReader, GGUFTensorInfo
from .projection import (
    LoRAFactors,
    apply_lora,
    compile_projection_factors,
    explicit_project_weight,
    left_contract_chunked,
    orthonormalize_directions,
)
from .quantization import GGMLQuantizationType, dequantize_rows, quantize_rows
from .tensor_source import GGUFWeightSource, TensorChunk

__all__ = [
    "GGMLQuantizationType",
    "GGUFReader",
    "GGUFTensorInfo",
    "GGUFWeightSource",
    "LoRAFactors",
    "TensorChunk",
    "apply_lora",
    "compile_projection_factors",
    "dequantize_rows",
    "explicit_project_weight",
    "left_contract_chunked",
    "orthonormalize_directions",
    "quantize_rows",
]
