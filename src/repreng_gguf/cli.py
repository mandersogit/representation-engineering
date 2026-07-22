"""Command-line interface for GGUF-native projection compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from safetensors.torch import load_file
import torch

from .adapter import ProjectionTarget, compile_projection_adapter
from .gguf_reader import GGUFReader
from .quantization import GGMLQuantizationType
from .validation import validate_lora_adapter

SUPPORTED_DEQUANTIZATION = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q6_K,
}


def _tensor_records(path: Path, pattern: str | None) -> list[dict[str, object]]:
    regex = re.compile(pattern) if pattern else None
    with GGUFReader(path) as reader:
        records = []
        for name, info in sorted(reader.tensors.items()):
            if regex and not regex.search(name):
                continue
            records.append(
                {
                    "name": name,
                    "shape": list(info.logical_shape),
                    "type": info.quantization.name,
                    "bytes": info.n_bytes,
                    "projection_supported": (
                        info.is_matrix and info.quantization in SUPPORTED_DEQUANTIZATION
                    ),
                }
            )
        return records


def command_inspect(args: argparse.Namespace) -> int:
    records = _tensor_records(args.model, args.pattern)
    if args.supported_only:
        records = [record for record in records if record["projection_supported"]]
    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    print(f"{'TYPE':<8} {'SHAPE':<24} {'MiB':>10}  TENSOR")
    for record in records:
        shape = "x".join(str(value) for value in record["shape"])
        marker = "*" if record["projection_supported"] else " "
        print(
            f"{record['type']:<8} {shape:<24} {record['bytes'] / 2**20:>10.2f} {marker} {record['name']}"
        )
    print("* projection compiler supports this rank-2 tensor type", file=sys.stderr)
    return 0


def command_compile(args: argparse.Namespace) -> int:
    direction_tensors = load_file(str(args.directions), device="cpu")
    if not direction_tensors:
        raise ValueError("direction archive contains no tensors")
    targets = [
        ProjectionTarget(name, tensor.to(torch.float32), args.strength)
        for name, tensor in sorted(direction_tensors.items())
    ]
    tensor_type = {
        "f16": GGMLQuantizationType.F16,
        "f32": GGMLQuantizationType.F32,
    }[args.output_type]
    compile_projection_adapter(
        args.model,
        args.output,
        targets,
        max_chunk_bytes=int(args.chunk_mib * 2**20),
        tensor_type=tensor_type,
        accumulation_dtype=torch.float32,
        name=args.name,
    )
    report = validate_lora_adapter(args.model, args.output)
    report.require_valid()
    print(
        f"wrote {args.output}: {report.tensor_pairs} tensor pairs, "
        f"rank(s)={sorted(report.ranks)}"
    )
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    report = validate_lora_adapter(args.model, args.adapter)
    payload = {
        "valid": report.valid,
        "tensor_pairs": report.tensor_pairs,
        "ranks": sorted(report.ranks),
        "errors": report.errors,
        "warnings": report.warnings,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("valid" if report.valid else "invalid")
        print(f"tensor pairs: {report.tensor_pairs}")
        print(f"ranks: {sorted(report.ranks)}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
    return 0 if report.valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repreng-gguf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="list GGUF tensors")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.add_argument("--pattern", help="regular expression matched against tensor names")
    inspect_parser.add_argument("--supported-only", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(handler=command_inspect)

    compile_parser = subparsers.add_parser(
        "compile", help="compile safetensors directions into a GGUF LoRA"
    )
    compile_parser.add_argument("model", type=Path)
    compile_parser.add_argument("directions", type=Path)
    compile_parser.add_argument("output", type=Path)
    compile_parser.add_argument("--strength", type=float, default=1.0)
    compile_parser.add_argument("--chunk-mib", type=float, default=64.0)
    compile_parser.add_argument("--output-type", choices=["f16", "f32"], default="f16")
    compile_parser.add_argument("--name", default="projection adapter")
    compile_parser.set_defaults(handler=command_compile)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a GGUF LoRA against its base model"
    )
    validate_parser.add_argument("model", type=Path)
    validate_parser.add_argument("adapter", type=Path)
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, NotImplementedError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
