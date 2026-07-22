"""Compile projection-derived LoRA factors into a llama.cpp GGUF adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .gguf_writer import GGUFWriter
from .projection import LoRAFactors
from .quantization import GGMLQuantizationType
from .tensor_source import GGUFWeightSource


@dataclass(frozen=True)
class ProjectionTarget:
    tensor_name: str
    directions: Tensor
    strength: float = 1.0


def write_lora_adapter(
    output_path: str | Path,
    *,
    architecture: str,
    factors: Mapping[str, LoRAFactors],
    tensor_type: GGMLQuantizationType = GGMLQuantizationType.F16,
    name: str = "projection adapter",
) -> None:
    """Write precomputed factors in llama.cpp's native GGUF-LoRA format.

    The factors already include intervention strength in B. Setting alpha equal
    to rank makes llama.cpp's standard alpha/rank multiplier equal to one.
    All tensor pairs must use the same rank because adapter.lora.alpha is global.
    """

    if not factors:
        raise ValueError("at least one LoRA tensor pair is required")
    ranks = {pair.rank for pair in factors.values()}
    if len(ranks) != 1:
        raise ValueError("all tensor pairs in a GGUF LoRA must use the same rank")
    rank = ranks.pop()

    writer = GGUFWriter()
    writer.add_string("general.type", "adapter")
    writer.add_string("general.architecture", architecture)
    writer.add_string("general.name", name)
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(rank))

    for base_name, pair in factors.items():
        writer.add_tensor(
            base_name + ".lora_a",
            pair.A.detach().cpu().to(torch.float32).numpy(),
            quantization=tensor_type,
        )
        writer.add_tensor(
            base_name + ".lora_b",
            pair.B.detach().cpu().to(torch.float32).numpy(),
            quantization=tensor_type,
        )
    writer.write(output_path)


def compile_projection_adapter(
    model_path: str | Path,
    output_path: str | Path,
    targets: list[ProjectionTarget],
    *,
    max_chunk_bytes: int,
    tensor_type: GGMLQuantizationType = GGMLQuantizationType.F16,
    accumulation_dtype: torch.dtype = torch.float32,
    name: str = "projection adapter",
) -> dict[str, LoRAFactors]:
    """Stream selected GGUF tensors, compile factors, and write an adapter."""

    if not targets:
        raise ValueError("at least one projection target is required")
    with GGUFWeightSource(model_path) as source:
        architecture = source.reader.metadata.get("general.architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("base GGUF does not declare general.architecture")
        factors: dict[str, LoRAFactors] = {}
        for target in targets:
            factors[target.tensor_name] = source.compile_projection_factors(
                target.tensor_name,
                target.directions,
                strength=target.strength,
                max_chunk_bytes=max_chunk_bytes,
                accumulation_dtype=accumulation_dtype,
            )

    write_lora_adapter(
        output_path,
        architecture=architecture,
        factors=factors,
        tensor_type=tensor_type,
        name=name,
    )
    return factors
