from __future__ import annotations

import pytest
import torch

from repreng_gguf.projection import (
    apply_lora,
    compile_projection_factors,
    explicit_project_weight,
    left_contract_chunked,
    orthonormalize_directions,
)


def seeded_tensor(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260722)
    return torch.randn(*shape, generator=generator, dtype=dtype)


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 17, 64, 1000])
def test_chunked_contraction_matches_dense(rank: int, rows_per_chunk: int) -> None:
    weight = seeded_tensor(97, 53)
    directions = orthonormalize_directions(seeded_tensor(97, rank))

    expected = directions.T @ weight
    actual = left_contract_chunked(
        directions,
        weight,
        rows_per_chunk=rows_per_chunk,
        accumulation_dtype=torch.float64,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0, 1.5])
def test_projection_lora_matches_explicit(rank: int, strength: float) -> None:
    weight = seeded_tensor(83, 47)
    directions = seeded_tensor(83, rank)

    factors = compile_projection_factors(
        weight,
        directions,
        strength=strength,
        rows_per_chunk=7,
        accumulation_dtype=torch.float64,
    )
    expected = explicit_project_weight(weight, directions, strength=strength)
    actual = apply_lora(weight, factors)

    assert factors.A.shape == (rank, 47)
    assert factors.B.shape == (83, rank)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_projection_removes_selected_subspace_at_full_strength() -> None:
    weight = seeded_tensor(71, 29)
    directions = orthonormalize_directions(seeded_tensor(71, 3))

    factors = compile_projection_factors(
        weight,
        directions,
        rows_per_chunk=11,
        orthonormalize=False,
        accumulation_dtype=torch.float64,
    )
    projected = apply_lora(weight, factors)

    residual_component = directions.T @ projected
    torch.testing.assert_close(
        residual_component,
        torch.zeros_like(residual_component),
        rtol=0,
        atol=1e-12,
    )


def test_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="output dimension"):
        left_contract_chunked(
            torch.randn(8),
            torch.randn(7, 4),
            rows_per_chunk=2,
        )


def test_rejects_dependent_directions() -> None:
    vector = seeded_tensor(16)
    dependent = torch.stack([vector, 2 * vector], dim=1)
    with pytest.raises(ValueError, match="linearly dependent"):
        orthonormalize_directions(dependent)
