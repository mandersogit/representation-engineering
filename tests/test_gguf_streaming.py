from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from repreng_gguf import (
    GGMLQuantizationType,
    GGUFReader,
    GGUFWeightSource,
    apply_lora,
    dequantize_rows,
    explicit_project_weight,
    quantize_rows,
)
from repreng_gguf.gguf_reader import GGUFValueType


@dataclass(frozen=True)
class FixtureTensor:
    name: str
    values: np.ndarray
    qtype: GGMLQuantizationType

    @property
    def raw_rows(self) -> np.ndarray:
        return quantize_rows(self.values, self.qtype)


class TinyGGUFWriter:
    """Test-only GGUF v3 writer with enough metadata variety to test skipping."""

    def __init__(self, *, alignment: int = 64):
        self.alignment = alignment
        self.metadata: list[tuple[str, GGUFValueType, object]] = [
            ("general.alignment", GGUFValueType.UINT32, alignment),
            ("general.architecture", GGUFValueType.STRING, "fixture"),
            ("fixture.float", GGUFValueType.FLOAT32, 1.25),
            (
                "fixture.array",
                GGUFValueType.ARRAY,
                (GGUFValueType.UINT16, [3, 5, 8]),
            ),
        ]
        self.tensors: list[FixtureTensor] = []

    def add(self, tensor: FixtureTensor) -> None:
        self.tensors.append(tensor)

    @staticmethod
    def _write_string(output: io.BytesIO, value: str) -> None:
        encoded = value.encode("utf-8")
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)

    def _write_value(self, output: io.BytesIO, value_type: GGUFValueType, value: object) -> None:
        scalar_formats = {
            GGUFValueType.UINT8: "B",
            GGUFValueType.INT8: "b",
            GGUFValueType.UINT16: "H",
            GGUFValueType.INT16: "h",
            GGUFValueType.UINT32: "I",
            GGUFValueType.INT32: "i",
            GGUFValueType.FLOAT32: "f",
            GGUFValueType.BOOL: "?",
            GGUFValueType.UINT64: "Q",
            GGUFValueType.INT64: "q",
            GGUFValueType.FLOAT64: "d",
        }
        if value_type == GGUFValueType.STRING:
            self._write_string(output, str(value))
            return
        if value_type == GGUFValueType.ARRAY:
            item_type, items = value  # type: ignore[misc]
            output.write(struct.pack("<I", int(item_type)))
            output.write(struct.pack("<Q", len(items)))
            for item in items:
                self._write_value(output, item_type, item)
            return
        output.write(struct.pack("<" + scalar_formats[value_type], value))

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return (value + alignment - 1) & ~(alignment - 1)

    def write(self, path: Path) -> None:
        relative_offsets: list[int] = []
        data_size = 0
        for tensor in self.tensors:
            data_size = self._align(data_size, self.alignment)
            relative_offsets.append(data_size)
            data_size += tensor.raw_rows.nbytes

        output = io.BytesIO()
        output.write(b"GGUF")
        output.write(struct.pack("<IQQ", 3, len(self.tensors), len(self.metadata)))

        for key, value_type, value in self.metadata:
            self._write_string(output, key)
            output.write(struct.pack("<I", int(value_type)))
            self._write_value(output, value_type, value)

        for tensor, relative_offset in zip(self.tensors, relative_offsets, strict=True):
            self._write_string(output, tensor.name)
            logical_shape = tensor.values.shape
            ggml_shape = tuple(reversed(logical_shape))
            output.write(struct.pack("<I", len(ggml_shape)))
            for dim in ggml_shape:
                output.write(struct.pack("<Q", dim))
            output.write(struct.pack("<IQ", int(tensor.qtype), relative_offset))

        data_offset = self._align(output.tell(), self.alignment)
        output.write(b"\x00" * (data_offset - output.tell()))
        for tensor, relative_offset in zip(self.tensors, relative_offsets, strict=True):
            desired = data_offset + relative_offset
            output.write(b"\x00" * (desired - output.tell()))
            output.write(tensor.raw_rows.tobytes(order="C"))

        path.write_bytes(output.getvalue())


@pytest.fixture
def tiny_gguf(tmp_path: Path) -> tuple[Path, dict[str, FixtureTensor]]:
    rng = np.random.default_rng(20260722)
    tensors = {
        "f16.weight": FixtureTensor(
            "f16.weight", rng.normal(size=(5, 11)).astype(np.float32), GGMLQuantizationType.F16
        ),
        "q8.weight": FixtureTensor(
            "q8.weight", rng.normal(size=(7, 64)).astype(np.float32), GGMLQuantizationType.Q8_0
        ),
        "q4.weight": FixtureTensor(
            "q4.weight", rng.normal(size=(9, 64)).astype(np.float32), GGMLQuantizationType.Q4_0
        ),
    }
    writer = TinyGGUFWriter()
    for tensor in tensors.values():
        writer.add(tensor)
    path = tmp_path / "tiny.gguf"
    writer.write(path)
    return path, tensors


def test_manual_q8_0_layout() -> None:
    scale = np.array([0.5], dtype=np.float16).view(np.uint8)
    quants = np.arange(-16, 16, dtype=np.int8)
    raw = np.concatenate([scale, quants.view(np.uint8)]).reshape(1, 34)
    actual = dequantize_rows(raw, GGMLQuantizationType.Q8_0, 32)
    expected = quants.astype(np.float32) * np.float32(0.5)
    np.testing.assert_array_equal(actual[0], expected)


def test_manual_q4_0_layout() -> None:
    scale = np.array([0.25], dtype=np.float16).view(np.uint8)
    low = np.arange(16, dtype=np.uint8)
    high = np.arange(15, -1, -1, dtype=np.uint8)
    packed = low | (high << np.uint8(4))
    raw = np.concatenate([scale, packed]).reshape(1, 18)
    actual = dequantize_rows(raw, GGMLQuantizationType.Q4_0, 32)
    expected_quants = np.concatenate([low, high]).astype(np.int8) - 8
    np.testing.assert_array_equal(actual[0], expected_quants.astype(np.float32) * 0.25)


def test_reader_parses_metadata_shapes_and_offsets(
    tiny_gguf: tuple[Path, dict[str, FixtureTensor]],
) -> None:
    path, tensors = tiny_gguf
    with GGUFReader(path) as reader:
        assert reader.version == 3
        assert reader.alignment == 64
        assert reader.metadata["general.architecture"] == "fixture"
        assert reader.metadata["fixture.array"] == [3, 5, 8]
        assert reader.data_offset % 64 == 0
        for name, tensor in tensors.items():
            info = reader.tensor_info(name)
            assert info.logical_shape == tensor.values.shape
            assert info.ggml_shape == tuple(reversed(tensor.values.shape))
            assert info.data_offset % 64 == 0
            assert info.n_bytes == tensor.raw_rows.nbytes


def test_raw_row_chunks_are_mmap_views(
    tiny_gguf: tuple[Path, dict[str, FixtureTensor]],
) -> None:
    path, tensors = tiny_gguf
    with GGUFReader(path) as reader:
        chunks = list(reader.iter_raw_rows("q8.weight", rows_per_chunk=3))
        assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 3), (3, 6), (6, 7)]
        combined = np.concatenate([chunk.data for chunk in chunks], axis=0)
        np.testing.assert_array_equal(combined, tensors["q8.weight"].raw_rows)
        assert all(isinstance(chunk.data.base, np.memmap) for chunk in chunks)


@pytest.mark.parametrize("name", ["f16.weight", "q8.weight", "q4.weight"])
@pytest.mark.parametrize("max_chunk_bytes", [1, 128, 1024, 1 << 20])
def test_streamed_left_contraction_matches_dense_reference(
    tiny_gguf: tuple[Path, dict[str, FixtureTensor]],
    name: str,
    max_chunk_bytes: int,
) -> None:
    path, tensors = tiny_gguf
    tensor = tensors[name]
    expected_weight = torch.from_numpy(
        dequantize_rows(tensor.raw_rows, tensor.qtype, tensor.values.shape[1])
    )
    generator = torch.Generator().manual_seed(1234)
    directions = torch.randn(tensor.values.shape[0], 3, generator=generator)

    with GGUFWeightSource(path) as source:
        actual = source.left_contract(
            name,
            directions,
            max_chunk_bytes=max_chunk_bytes,
            accumulation_dtype=torch.float64,
        )

    expected = directions.to(torch.float64).T @ expected_weight.to(torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_streamed_projection_factors_match_explicit_quantized_projection(
    tiny_gguf: tuple[Path, dict[str, FixtureTensor]],
) -> None:
    path, tensors = tiny_gguf
    tensor = tensors["q4.weight"]
    quantized_weight = torch.from_numpy(
        dequantize_rows(tensor.raw_rows, tensor.qtype, tensor.values.shape[1])
    ).to(torch.float64)
    generator = torch.Generator().manual_seed(99)
    directions = torch.randn(tensor.values.shape[0], 4, generator=generator, dtype=torch.float64)

    with GGUFWeightSource(path) as source:
        factors = source.compile_projection_factors(
            "q4.weight",
            directions,
            strength=0.75,
            max_chunk_bytes=256,
            accumulation_dtype=torch.float64,
        )

    actual = apply_lora(quantized_weight, factors)
    expected = explicit_project_weight(quantized_weight, directions, strength=0.75)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
