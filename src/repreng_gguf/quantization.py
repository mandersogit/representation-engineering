"""Minimal GGML quantization support needed by the streaming prototype.

The numeric layouts and algorithms mirror llama.cpp's MIT-licensed gguf-py
implementation. The module intentionally starts with the common dense tensor
formats needed to validate the architecture; K-quants are added separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import prod
from typing import Sequence

import numpy as np


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30
    MXFP4 = 39
    NVFP4 = 40


@dataclass(frozen=True)
class QuantizationSpec:
    block_size: int
    type_size: int


QUANTIZATION_SPECS: dict[GGMLQuantizationType, QuantizationSpec] = {
    GGMLQuantizationType.F32: QuantizationSpec(1, 4),
    GGMLQuantizationType.F16: QuantizationSpec(1, 2),
    GGMLQuantizationType.Q4_0: QuantizationSpec(32, 18),
    GGMLQuantizationType.Q4_1: QuantizationSpec(32, 20),
    GGMLQuantizationType.Q5_0: QuantizationSpec(32, 22),
    GGMLQuantizationType.Q5_1: QuantizationSpec(32, 24),
    GGMLQuantizationType.Q8_0: QuantizationSpec(32, 34),
    GGMLQuantizationType.Q8_1: QuantizationSpec(32, 40),
    GGMLQuantizationType.Q2_K: QuantizationSpec(256, 84),
    GGMLQuantizationType.Q3_K: QuantizationSpec(256, 110),
    GGMLQuantizationType.Q4_K: QuantizationSpec(256, 144),
    GGMLQuantizationType.Q5_K: QuantizationSpec(256, 176),
    GGMLQuantizationType.Q6_K: QuantizationSpec(256, 210),
    GGMLQuantizationType.Q8_K: QuantizationSpec(256, 292),
    GGMLQuantizationType.BF16: QuantizationSpec(1, 2),
}


def quantization_spec(qtype: GGMLQuantizationType) -> QuantizationSpec:
    try:
        return QUANTIZATION_SPECS[qtype]
    except KeyError as exc:
        raise NotImplementedError(f"unsupported GGML quantization type: {qtype.name}") from exc


def row_storage_bytes(width: int, qtype: GGMLQuantizationType) -> int:
    spec = quantization_spec(qtype)
    if width % spec.block_size:
        raise ValueError(
            f"row width {width} is not divisible by {qtype.name} block size "
            f"{spec.block_size}"
        )
    return width // spec.block_size * spec.type_size


def tensor_storage_bytes(shape: Sequence[int], qtype: GGMLQuantizationType) -> int:
    if not shape:
        raise ValueError("tensor shape cannot be empty")
    return prod(shape[:-1]) * row_storage_bytes(shape[-1], qtype)


def _require_raw_rows(raw_rows: np.ndarray, row_bytes: int) -> np.ndarray:
    rows = np.asarray(raw_rows, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != row_bytes:
        raise ValueError(f"expected raw byte rows with shape (n, {row_bytes}), got {rows.shape}")
    return rows


def _dequantize_q8_0(raw_rows: np.ndarray, width: int) -> np.ndarray:
    row_bytes = row_storage_bytes(width, GGMLQuantizationType.Q8_0)
    rows = _require_raw_rows(raw_rows, row_bytes)
    blocks = rows.reshape(-1, 34)
    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    quants = blocks[:, 2:].view(np.int8).astype(np.float32)
    values = quants * scales
    return values.reshape(rows.shape[0], width)


def _dequantize_q4_0(raw_rows: np.ndarray, width: int) -> np.ndarray:
    row_bytes = row_storage_bytes(width, GGMLQuantizationType.Q4_0)
    rows = _require_raw_rows(raw_rows, row_bytes)
    blocks = rows.reshape(-1, 18)
    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    packed = blocks[:, 2:]
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    quants = np.concatenate([low, high], axis=1).astype(np.int8) - np.int8(8)
    values = scales * quants.astype(np.float32)
    return values.reshape(rows.shape[0], width)


def _unpack_q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the eight 6-bit scales and minima in a Q4_K block."""

    if scales.ndim != 2 or scales.shape[1] != 12:
        raise ValueError(f"expected Q4_K scales with shape (n, 12), got {scales.shape}")
    grouped = scales.reshape(scales.shape[0], 3, 4)
    low_scales, low_mins, high_bits = np.split(grouped, 3, axis=1)
    decoded_scales = np.concatenate(
        [
            low_scales & np.uint8(0x3F),
            (high_bits & np.uint8(0x0F))
            | ((low_scales >> np.uint8(2)) & np.uint8(0x30)),
        ],
        axis=-1,
    )
    decoded_mins = np.concatenate(
        [
            low_mins & np.uint8(0x3F),
            (high_bits >> np.uint8(4))
            | ((low_mins >> np.uint8(2)) & np.uint8(0x30)),
        ],
        axis=-1,
    )
    return (
        decoded_scales.reshape(scales.shape[0], 8),
        decoded_mins.reshape(scales.shape[0], 8),
    )


def _dequantize_q4_k(raw_rows: np.ndarray, width: int) -> np.ndarray:
    row_bytes = row_storage_bytes(width, GGMLQuantizationType.Q4_K)
    rows = _require_raw_rows(raw_rows, row_bytes)
    blocks = rows.reshape(-1, 144)

    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    min_scales = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    subblock_scales, subblock_mins = _unpack_q4_k_scale_min(blocks[:, 4:16])

    effective_scales = (scales * subblock_scales.astype(np.float32)).reshape(-1, 8, 1)
    effective_mins = (min_scales * subblock_mins.astype(np.float32)).reshape(-1, 8, 1)

    packed = blocks[:, 16:].reshape(-1, 4, 1, 32)
    quants = packed >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    quants = (quants & np.uint8(0x0F)).reshape(-1, 8, 32).astype(np.float32)
    values = effective_scales * quants - effective_mins
    return values.reshape(rows.shape[0], width)


def dequantize_rows(
    raw_rows: np.ndarray,
    qtype: GGMLQuantizationType,
    width: int,
) -> np.ndarray:
    """Dequantize complete logical rows to an FP32 matrix."""

    row_bytes = row_storage_bytes(width, qtype)
    rows = _require_raw_rows(raw_rows, row_bytes)

    if qtype == GGMLQuantizationType.F32:
        return (
            rows.copy()
            .reshape(-1)
            .view("<f4")
            .reshape(rows.shape[0], width)
            .astype(np.float32, copy=False)
        )
    if qtype == GGMLQuantizationType.F16:
        return rows.copy().reshape(-1).view("<f2").reshape(rows.shape[0], width).astype(np.float32)
    if qtype == GGMLQuantizationType.BF16:
        words = rows.copy().reshape(-1).view("<u2").astype(np.uint32)
        return (words << np.uint32(16)).view(np.float32).reshape(rows.shape[0], width)
    if qtype == GGMLQuantizationType.Q8_0:
        return _dequantize_q8_0(rows, width)
    if qtype == GGMLQuantizationType.Q4_0:
        return _dequantize_q4_0(rows, width)
    if qtype == GGMLQuantizationType.Q4_K:
        return _dequantize_q4_k(rows, width)
    raise NotImplementedError(f"dequantization is not implemented for {qtype.name}")


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    floored = np.floor(absolute)
    rounded = floored + np.floor(2 * (absolute - floored))
    return np.sign(values) * rounded


def quantize_rows(values: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    """Reference quantizer used only for deterministic tests and fixtures."""

    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("values must be a rank-2 row matrix")
    width = rows.shape[1]

    if qtype == GGMLQuantizationType.F32:
        return rows.astype("<f4", copy=False).view(np.uint8).reshape(rows.shape[0], -1).copy()
    if qtype == GGMLQuantizationType.F16:
        return rows.astype("<f2").view(np.uint8).reshape(rows.shape[0], -1)
    if qtype == GGMLQuantizationType.BF16:
        bits = rows.view(np.uint32)
        rounded = (bits.astype(np.uint64) + (0x7FFF + ((bits >> 16) & 1))) >> 16
        return rounded.astype("<u2").view(np.uint8).reshape(rows.shape[0], -1)

    spec = quantization_spec(qtype)
    if width % spec.block_size:
        raise ValueError(f"row width must be divisible by {spec.block_size}")
    blocks = rows.reshape(-1, spec.block_size)

    if qtype == GGMLQuantizationType.Q8_0:
        scale = np.max(np.abs(blocks), axis=1, keepdims=True) / np.float32(127)
        reciprocal = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
        quants = _round_away_from_zero(blocks * reciprocal).astype(np.int8)
        packed = np.concatenate(
            [scale.astype(np.float16).view(np.uint8), quants.view(np.uint8)], axis=1
        )
        return packed.reshape(rows.shape[0], -1)

    if qtype == GGMLQuantizationType.Q4_0:
        max_indices = np.argmax(np.abs(blocks), axis=1, keepdims=True)
        signed_max = np.take_along_axis(blocks, max_indices, axis=1)
        scale = signed_max / np.float32(-8)
        reciprocal = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
        quants = np.trunc(blocks * reciprocal + np.float32(8.5)).astype(np.uint8)
        quants = np.clip(quants, 0, 15)
        packed_quants = quants[:, :16] | (quants[:, 16:] << np.uint8(4))
        packed = np.concatenate(
            [scale.astype(np.float16).view(np.uint8), packed_quants], axis=1
        )
        return packed.reshape(rows.shape[0], -1)

    raise NotImplementedError(f"test quantization is not implemented for {qtype.name}")
