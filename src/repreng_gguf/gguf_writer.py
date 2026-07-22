"""Small GGUF v3 writer for representation-engineering artifacts.

The prototype only needs scalar metadata and dense F32/F16 tensors. Keeping the
writer narrow makes the adapter format auditable while the base model reader
continues to support quantized tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct

import numpy as np

from .gguf_reader import GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GGUFValueType
from .quantization import GGMLQuantizationType


@dataclass(frozen=True)
class DenseGGUFTensor:
    name: str
    values: np.ndarray
    quantization: GGMLQuantizationType

    def encoded(self) -> bytes:
        values = np.asarray(self.values)
        if self.quantization == GGMLQuantizationType.F32:
            return values.astype("<f4", copy=False).tobytes(order="C")
        if self.quantization == GGMLQuantizationType.F16:
            return values.astype("<f2").tobytes(order="C")
        raise ValueError("artifact writer only supports F32 and F16 tensors")


class GGUFWriter:
    """Write a compact GGUF v3 file with dense artifact tensors."""

    def __init__(self, *, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a non-zero power of two")
        self.alignment = alignment
        self.metadata: dict[str, tuple[GGUFValueType, object]] = {}
        self.tensors: list[DenseGGUFTensor] = []

    def add_string(self, key: str, value: str) -> None:
        self._add_metadata(key, GGUFValueType.STRING, value)

    def add_uint32(self, key: str, value: int) -> None:
        self._add_metadata(key, GGUFValueType.UINT32, value)

    def add_float32(self, key: str, value: float) -> None:
        self._add_metadata(key, GGUFValueType.FLOAT32, value)

    def _add_metadata(self, key: str, value_type: GGUFValueType, value: object) -> None:
        if key in self.metadata:
            raise ValueError(f"duplicate GGUF metadata key {key!r}")
        self.metadata[key] = (value_type, value)

    def add_tensor(
        self,
        name: str,
        values: np.ndarray,
        *,
        quantization: GGMLQuantizationType = GGMLQuantizationType.F16,
    ) -> None:
        if any(tensor.name == name for tensor in self.tensors):
            raise ValueError(f"duplicate GGUF tensor name {name!r}")
        array = np.asarray(values)
        if not array.shape:
            raise ValueError("GGUF tensors must have at least one dimension")
        self.tensors.append(DenseGGUFTensor(name, array, quantization))

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return (value + alignment - 1) & ~(alignment - 1)

    @staticmethod
    def _write_string(output: io.BytesIO, value: str) -> None:
        encoded = value.encode("utf-8")
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)

    @classmethod
    def _write_value(
        cls, output: io.BytesIO, value_type: GGUFValueType, value: object
    ) -> None:
        if value_type == GGUFValueType.STRING:
            cls._write_string(output, str(value))
            return
        if value_type == GGUFValueType.UINT32:
            output.write(struct.pack("<I", int(value)))
            return
        if value_type == GGUFValueType.FLOAT32:
            output.write(struct.pack("<f", float(value)))
            return
        raise ValueError(f"unsupported artifact metadata type {value_type.name}")

    def write(self, path: str | Path) -> None:
        encoded_tensors = [tensor.encoded() for tensor in self.tensors]
        relative_offsets: list[int] = []
        data_size = 0
        for encoded in encoded_tensors:
            data_size = self._align(data_size, self.alignment)
            relative_offsets.append(data_size)
            data_size += len(encoded)

        output = io.BytesIO()
        output.write(GGUF_MAGIC)
        output.write(
            struct.pack(
                "<IQQ", GGUF_VERSION, len(self.tensors), len(self.metadata)
            )
        )

        for key, (value_type, value) in self.metadata.items():
            self._write_string(output, key)
            output.write(struct.pack("<I", int(value_type)))
            self._write_value(output, value_type, value)

        for tensor, relative_offset in zip(
            self.tensors, relative_offsets, strict=True
        ):
            self._write_string(output, tensor.name)
            ggml_shape = tuple(reversed(tensor.values.shape))
            output.write(struct.pack("<I", len(ggml_shape)))
            for dimension in ggml_shape:
                output.write(struct.pack("<Q", int(dimension)))
            output.write(
                struct.pack(
                    "<IQ", int(tensor.quantization), relative_offset
                )
            )

        data_offset = self._align(output.tell(), self.alignment)
        output.write(b"\x00" * (data_offset - output.tell()))
        for encoded, relative_offset in zip(
            encoded_tensors, relative_offsets, strict=True
        ):
            desired_offset = data_offset + relative_offset
            output.write(b"\x00" * (desired_offset - output.tell()))
            output.write(encoded)

        Path(path).write_bytes(output.getvalue())
