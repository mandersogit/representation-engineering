#!/usr/bin/env python3
"""Benchmark bounded-memory contraction over a sparse Q4_K GGUF tensor.

The fixture is a valid GGUF whose tensor data is a sparse zero-filled file.
It lets the experiment exercise mmap, row slicing, Q4_K dequantization, and
R^T W accumulation at sizes where a dense FP32 reconstruction would be much
larger than the on-disk tensor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

import torch

from repreng_gguf.gguf_reader import GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION
from repreng_gguf.quantization import GGMLQuantizationType, tensor_storage_bytes
from repreng_gguf.tensor_source import GGUFWeightSource

TENSOR_NAME = "blk.0.attn_output.weight"


def _write_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def create_sparse_fixture(path: Path, rows: int, width: int) -> dict[str, int]:
    if width % 256:
        raise ValueError("Q4_K width must be divisible by 256")
    metadata = [
        ("general.alignment", 4, struct.pack("<I", GGUF_DEFAULT_ALIGNMENT)),
        ("general.type", 8, "model"),
        ("general.architecture", 8, "qwen2"),
        ("general.name", 8, "sparse Q4_K benchmark fixture"),
    ]
    with path.open("wb") as handle:
        handle.write(GGUF_MAGIC)
        handle.write(struct.pack("<IQQ", GGUF_VERSION, 1, len(metadata)))
        for key, value_type, value in metadata:
            _write_string(handle, key)
            handle.write(struct.pack("<I", value_type))
            if value_type == 8:
                _write_string(handle, str(value))
            else:
                handle.write(value)

        _write_string(handle, TENSOR_NAME)
        handle.write(struct.pack("<I", 2))
        handle.write(struct.pack("<QQ", width, rows))
        handle.write(struct.pack("<IQ", int(GGMLQuantizationType.Q4_K), 0))

        offset = handle.tell()
        data_offset = (offset + GGUF_DEFAULT_ALIGNMENT - 1) & ~(
            GGUF_DEFAULT_ALIGNMENT - 1
        )
        handle.write(b"\x00" * (data_offset - offset))
        storage_bytes = tensor_storage_bytes(
            (rows, width), GGMLQuantizationType.Q4_K
        )
        handle.truncate(data_offset + storage_bytes)
    return {
        "data_offset": data_offset,
        "storage_bytes": storage_bytes,
        "dense_fp32_bytes": rows * width * 4,
    }


def current_rss_kib() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")


def worker(args: argparse.Namespace) -> int:
    generator = torch.Generator().manual_seed(args.seed)
    directions = torch.randn(args.rows, args.rank, generator=generator)
    rss_before = current_rss_kib()
    started = time.perf_counter()
    with GGUFWeightSource(args.fixture) as source:
        result = source.left_contract(
            TENSOR_NAME,
            directions,
            max_chunk_bytes=int(args.chunk_mib * 2**20),
            accumulation_dtype=torch.float32,
        )
    elapsed = time.perf_counter() - started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "chunk_mib": args.chunk_mib,
        "rows": args.rows,
        "width": args.width,
        "rank": args.rank,
        "elapsed_seconds": elapsed,
        "rss_before_mib": rss_before / 1024,
        "peak_rss_mib": peak_rss_kib / 1024,
        "peak_increment_mib": max(0, peak_rss_kib - rss_before) / 1024,
        "result_shape": list(result.shape),
        "result_norm": float(torch.linalg.vector_norm(result)),
    }
    print(json.dumps(payload))
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture = create_sparse_fixture(args.fixture, args.rows, args.width)
    results = []
    for chunk_mib in args.chunk_mib:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--fixture",
            str(args.fixture),
            "--rows",
            str(args.rows),
            "--width",
            str(args.width),
            "--rank",
            str(args.rank),
            "--seed",
            str(args.seed),
            "--chunk-mib",
            str(chunk_mib),
        ]
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))
    print(
        json.dumps(
            {
                "fixture": str(args.fixture),
                "storage_mib": fixture["storage_bytes"] / 2**20,
                "dense_fp32_mib": fixture["dense_fp32_bytes"] / 2**20,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fixture", type=Path, default=Path("artifacts/sparse-q4k-benchmark.gguf")
    )
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--chunk-mib", type=float, nargs="+", default=[1, 8, 64])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker:
        if len(args.chunk_mib) != 1:
            raise ValueError("worker accepts exactly one chunk size")
        args.chunk_mib = args.chunk_mib[0]
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
