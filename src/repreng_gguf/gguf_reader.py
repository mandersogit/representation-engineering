"""Small, mmap-backed GGUF v3 reader for tensor-directory access.

This is deliberately not a general model loader. It parses enough of the
self-describing GGUF container to locate arbitrary tensors and expose their raw
row bytes without copying or constructing a Transformers model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import prod
from pathlib import Path
import struct
from typing import Any, Iterator

import numpy as np

from .quantization import (
    GGMLQuantizationType,
    row_storage_bytes,
    tensor_storage_bytes,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: dict[GGUFValueType, str] = {
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


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    logical_shape: tuple[int, ...]
    ggml_shape: tuple[int, ...]
    quantization: GGMLQuantizationType
    relative_offset: int
    data_offset: int
    n_elements: int
    n_bytes: int

    @property
    def is_matrix(self) -> bool:
        return len(self.logical_shape) == 2

    @property
    def out_features(self) -> int:
        if not self.is_matrix:
            raise ValueError(f"tensor {self.name!r} is not a matrix")
        return self.logical_shape[0]

    @property
    def in_features(self) -> int:
        if not self.is_matrix:
            raise ValueError(f"tensor {self.name!r} is not a matrix")
        return self.logical_shape[1]

    @property
    def row_bytes(self) -> int:
        return row_storage_bytes(self.in_features, self.quantization)


@dataclass(frozen=True)
class RawTensorRows:
    start: int
    end: int
    data: np.ndarray


class GGUFReader:
    """Parse GGUF metadata and expose tensor data through an mmap."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = np.memmap(self.path, mode="r", dtype=np.uint8)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self.version: int
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.data_offset: int
        self._parse()

    def close(self) -> None:
        mmap_obj = getattr(self._data, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def __enter__(self) -> GGUFReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _unpack_from(self, fmt: str, offset: int) -> tuple[Any, int]:
        full_fmt = "<" + fmt
        size = struct.calcsize(full_fmt)
        if offset + size > self._data.size:
            raise ValueError("unexpected end of GGUF file")
        return struct.unpack_from(full_fmt, self._data, offset)[0], offset + size

    def _read_string(self, offset: int) -> tuple[str, int]:
        length, offset = self._unpack_from("Q", offset)
        end = offset + int(length)
        if end > self._data.size:
            raise ValueError("GGUF string extends beyond end of file")
        try:
            value = bytes(self._data[offset:end]).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GGUF contains invalid UTF-8") from exc
        return value, end

    def _read_value(self, value_type: GGUFValueType, offset: int) -> tuple[Any, int]:
        if value_type == GGUFValueType.STRING:
            return self._read_string(offset)
        if value_type == GGUFValueType.ARRAY:
            raw_item_type, offset = self._unpack_from("I", offset)
            item_type = GGUFValueType(raw_item_type)
            length, offset = self._unpack_from("Q", offset)
            values: list[Any] = []
            for _ in range(int(length)):
                item, offset = self._read_value(item_type, offset)
                values.append(item)
            return values, offset
        try:
            fmt = _SCALAR_FORMATS[value_type]
        except KeyError as exc:
            raise ValueError(f"unsupported GGUF metadata value type {value_type}") from exc
        return self._unpack_from(fmt, offset)

    def _parse(self) -> None:
        if self._data.size < 24 or bytes(self._data[:4]) != GGUF_MAGIC:
            raise ValueError("invalid GGUF magic")
        offset = 4
        self.version, offset = self._unpack_from("I", offset)
        if self.version not in (2, GGUF_VERSION):
            raise ValueError(f"unsupported GGUF version {self.version}")
        tensor_count, offset = self._unpack_from("Q", offset)
        metadata_count, offset = self._unpack_from("Q", offset)

        for _ in range(int(metadata_count)):
            key, offset = self._read_string(offset)
            raw_type, offset = self._unpack_from("I", offset)
            value, offset = self._read_value(GGUFValueType(raw_type), offset)
            if key in self.metadata:
                raise ValueError(f"duplicate GGUF metadata key {key!r}")
            self.metadata[key] = value

        tensor_entries: list[tuple[str, tuple[int, ...], GGMLQuantizationType, int]] = []
        for _ in range(int(tensor_count)):
            name, offset = self._read_string(offset)
            n_dims, offset = self._unpack_from("I", offset)
            ggml_dims: list[int] = []
            for _ in range(int(n_dims)):
                dim, offset = self._unpack_from("Q", offset)
                ggml_dims.append(int(dim))
            raw_qtype, offset = self._unpack_from("I", offset)
            relative_offset, offset = self._unpack_from("Q", offset)
            tensor_entries.append(
                (name, tuple(ggml_dims), GGMLQuantizationType(raw_qtype), int(relative_offset))
            )

        alignment = self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment {alignment!r}")
        self.alignment = alignment
        self.data_offset = (offset + alignment - 1) & ~(alignment - 1)

        for name, ggml_shape, qtype, relative_offset in tensor_entries:
            if name in self.tensors:
                raise ValueError(f"duplicate GGUF tensor name {name!r}")
            logical_shape = tuple(reversed(ggml_shape))
            n_elements = prod(logical_shape)
            n_bytes = tensor_storage_bytes(logical_shape, qtype)
            data_offset = self.data_offset + relative_offset
            if data_offset + n_bytes > self._data.size:
                raise ValueError(f"tensor {name!r} extends beyond end of GGUF file")
            self.tensors[name] = GGUFTensorInfo(
                name=name,
                logical_shape=logical_shape,
                ggml_shape=ggml_shape,
                quantization=qtype,
                relative_offset=relative_offset,
                data_offset=data_offset,
                n_elements=n_elements,
                n_bytes=n_bytes,
            )

    def tensor_info(self, name: str) -> GGUFTensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def iter_raw_rows(self, name: str, *, rows_per_chunk: int) -> Iterator[RawTensorRows]:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        info = self.tensor_info(name)
        if not info.is_matrix:
            raise ValueError(f"tensor {name!r} must be rank 2 for row streaming")
        for start in range(0, info.out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, info.out_features)
            byte_start = info.data_offset + start * info.row_bytes
            byte_end = info.data_offset + end * info.row_bytes
            view = self._data[byte_start:byte_end].reshape(end - start, info.row_bytes)
            yield RawTensorRows(start=start, end=end, data=view)
