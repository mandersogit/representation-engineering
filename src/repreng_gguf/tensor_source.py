"""Memory-bounded tensor operations over GGUF model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from .gguf_reader import GGUFReader, GGUFTensorInfo
from .projection import LoRAFactors, orthonormalize_directions
from .quantization import dequantize_rows


@dataclass(frozen=True)
class TensorChunk:
    start: int
    end: int
    values: Tensor


class GGUFWeightSource:
    """Expose selected GGUF tensors without reconstructing the full model."""

    def __init__(self, path: str | Path):
        self.reader = GGUFReader(path)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> GGUFWeightSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def tensor_info(self, name: str) -> GGUFTensorInfo:
        return self.reader.tensor_info(name)

    @staticmethod
    def _rows_per_chunk(info: GGUFTensorInfo, max_chunk_bytes: int) -> int:
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        working_bytes_per_row = info.row_bytes + info.in_features * 4
        return max(1, max_chunk_bytes // working_bytes_per_row)

    def iter_dequantized_chunks(
        self,
        name: str,
        *,
        max_chunk_bytes: int,
    ) -> Iterator[TensorChunk]:
        info = self.tensor_info(name)
        rows_per_chunk = self._rows_per_chunk(info, max_chunk_bytes)
        for raw in self.reader.iter_raw_rows(name, rows_per_chunk=rows_per_chunk):
            values = dequantize_rows(raw.data, info.quantization, info.in_features)
            yield TensorChunk(raw.start, raw.end, torch.from_numpy(values))

    def left_contract(
        self,
        name: str,
        directions: Tensor,
        *,
        max_chunk_bytes: int,
        accumulation_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Compute ``directions.T @ W_q`` with bounded working memory."""

        info = self.tensor_info(name)
        if directions.ndim == 1:
            basis = directions[:, None]
        elif directions.ndim == 2:
            basis = directions
        else:
            raise ValueError("directions must have shape (out_features,) or (out_features, rank)")
        if basis.shape[0] != info.out_features:
            raise ValueError(
                f"direction output dimension {basis.shape[0]} does not match "
                f"tensor out_features {info.out_features}"
            )
        if basis.device.type != "cpu":
            raise ValueError("the prototype GGUF backend currently requires CPU directions")

        basis = basis.to(dtype=accumulation_dtype)
        result = torch.zeros((basis.shape[1], info.in_features), dtype=accumulation_dtype)
        for chunk in self.iter_dequantized_chunks(name, max_chunk_bytes=max_chunk_bytes):
            values = chunk.values.to(dtype=accumulation_dtype)
            result.add_(basis[chunk.start:chunk.end].T @ values)
        return result

    def compile_projection_factors(
        self,
        name: str,
        directions: Tensor,
        *,
        strength: float = 1.0,
        max_chunk_bytes: int,
        orthonormalize: bool = True,
        accumulation_dtype: torch.dtype = torch.float32,
    ) -> LoRAFactors:
        basis = orthonormalize_directions(directions) if orthonormalize else (
            directions[:, None] if directions.ndim == 1 else directions
        )
        a = self.left_contract(
            name,
            basis,
            max_chunk_bytes=max_chunk_bytes,
            accumulation_dtype=accumulation_dtype,
        )
        b = (-strength * basis).to(dtype=accumulation_dtype)
        return LoRAFactors(A=a, B=b)
