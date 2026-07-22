"""Projection-to-LoRA linear algebra.

The production GGUF implementation only needs to provide a memory-bounded
left contraction ``R.T @ W``. This module contains the format-independent
mathematics and a dense, row-chunked reference implementation used to verify
shape, orientation, and accumulation semantics before GGUF I/O is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor


@dataclass(frozen=True)
class LoRAFactors:
    """LoRA factors representing ``delta_weight = B @ A``.

    ``A`` has shape ``(rank, in_features)`` and ``B`` has shape
    ``(out_features, rank)``.
    """

    A: Tensor
    B: Tensor

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("LoRA factors must both be rank-2 tensors")
        if self.B.shape[1] != self.A.shape[0]:
            raise ValueError(
                "LoRA rank mismatch: B.shape[1] must equal A.shape[0]"
            )

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def delta_weight(self) -> Tensor:
        return self.B @ self.A


def _as_direction_matrix(directions: Tensor) -> Tensor:
    """Return directions as ``(out_features, rank)``."""

    if directions.ndim == 1:
        return directions[:, None]
    if directions.ndim == 2:
        return directions
    raise ValueError("directions must have shape (out_features,) or (out_features, rank)")


def orthonormalize_directions(directions: Tensor) -> Tensor:
    """Return an orthonormal basis spanning the supplied directions.

    Direction columns are reduced with QR. Linearly dependent directions are
    rejected rather than silently producing an ill-defined projection rank.
    """

    matrix = _as_direction_matrix(directions)
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("direction rank cannot exceed out_features")
    if not torch.is_floating_point(matrix):
        matrix = matrix.to(torch.float32)

    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    tolerance = torch.finfo(matrix.dtype).eps * max(matrix.shape) * torch.max(diagonal)
    if torch.any(diagonal <= tolerance):
        raise ValueError("directions are linearly dependent or numerically singular")
    return q


def _iter_row_ranges(row_count: int, rows_per_chunk: int) -> Iterator[tuple[int, int]]:
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    for start in range(0, row_count, rows_per_chunk):
        yield start, min(start + rows_per_chunk, row_count)


def left_contract_chunked(
    directions: Tensor,
    weight: Tensor,
    *,
    rows_per_chunk: int,
    accumulation_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute ``directions.T @ weight`` using bounded row chunks.

    This reference implementation accepts a dense tensor but deliberately
    touches it only through row slices. ``GGUFWeightSource`` will later replace
    each slice with a lazily dequantized GGUF row range while retaining exactly
    these accumulation semantics.
    """

    basis = _as_direction_matrix(directions)
    if weight.ndim != 2:
        raise ValueError("weight must have shape (out_features, in_features)")
    if basis.shape[0] != weight.shape[0]:
        raise ValueError(
            "direction output dimension must equal weight out_features: "
            f"{basis.shape[0]} != {weight.shape[0]}"
        )

    result = torch.zeros(
        (basis.shape[1], weight.shape[1]),
        dtype=accumulation_dtype,
        device=weight.device,
    )
    basis = basis.to(device=weight.device, dtype=accumulation_dtype)

    for start, end in _iter_row_ranges(weight.shape[0], rows_per_chunk):
        weight_chunk = weight[start:end].to(dtype=accumulation_dtype)
        result.add_(basis[start:end].T @ weight_chunk)
    return result


def compile_projection_factors(
    weight: Tensor,
    directions: Tensor,
    *,
    strength: float = 1.0,
    rows_per_chunk: int | None = None,
    orthonormalize: bool = True,
    accumulation_dtype: torch.dtype = torch.float32,
) -> LoRAFactors:
    """Compile a residual-space projection into LoRA factors.

    For orthonormal direction columns ``R`` this returns::

        A = R.T @ W
        B = -strength * R

    such that ``W + B @ A == (I - strength * R @ R.T) @ W``.
    """

    basis = (
        orthonormalize_directions(directions)
        if orthonormalize
        else _as_direction_matrix(directions)
    )
    if basis.shape[0] != weight.shape[0]:
        raise ValueError("directions and weight have incompatible output dimensions")

    if rows_per_chunk is None:
        a = basis.to(accumulation_dtype).T @ weight.to(accumulation_dtype)
    else:
        a = left_contract_chunked(
            basis,
            weight,
            rows_per_chunk=rows_per_chunk,
            accumulation_dtype=accumulation_dtype,
        )
    b = (-strength * basis).to(accumulation_dtype)
    return LoRAFactors(A=a, B=b)


def explicit_project_weight(
    weight: Tensor,
    directions: Tensor,
    *,
    strength: float = 1.0,
    orthonormalize: bool = True,
) -> Tensor:
    """Materialize the explicitly projected reference weight matrix."""

    basis = (
        orthonormalize_directions(directions)
        if orthonormalize
        else _as_direction_matrix(directions)
    )
    basis = basis.to(device=weight.device, dtype=weight.dtype)
    return weight - strength * basis @ (basis.T @ weight)


def apply_lora(weight: Tensor, factors: LoRAFactors) -> Tensor:
    """Materialize ``weight + B @ A`` for validation."""

    return weight + factors.B.to(weight) @ factors.A.to(weight)
