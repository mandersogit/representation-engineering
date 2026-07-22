"""Structural validation for llama.cpp GGUF LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .gguf_reader import GGUFReader, GGUFTensorInfo


@dataclass
class AdapterValidationReport:
    base_path: Path
    adapter_path: Path
    tensor_pairs: int = 0
    ranks: set[int] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("invalid GGUF LoRA adapter:\n- " + "\n- ".join(self.errors))


def validate_lora_adapter(
    base_path: str | Path,
    adapter_path: str | Path,
) -> AdapterValidationReport:
    """Mirror the format and shape checks performed by llama.cpp's loader."""

    report = AdapterValidationReport(Path(base_path), Path(adapter_path))
    with GGUFReader(base_path) as base, GGUFReader(adapter_path) as adapter:
        if adapter.metadata.get("general.type") != "adapter":
            report.errors.append("general.type must be 'adapter'")
        if adapter.metadata.get("adapter.type") != "lora":
            report.errors.append("adapter.type must be 'lora'")

        base_arch = base.metadata.get("general.architecture")
        adapter_arch = adapter.metadata.get("general.architecture")
        if not isinstance(base_arch, str) or not base_arch:
            report.errors.append("base model does not declare general.architecture")
        if adapter_arch != base_arch:
            report.errors.append(
                f"architecture mismatch: base={base_arch!r}, adapter={adapter_arch!r}"
            )

        pairs: dict[str, dict[str, GGUFTensorInfo]] = {}
        for name, info in adapter.tensors.items():
            if name.endswith(".lora_a"):
                base_name, side = name.removesuffix(".lora_a"), "a"
            elif name.endswith(".lora_b"):
                base_name, side = name.removesuffix(".lora_b"), "b"
            else:
                report.errors.append(f"unexpected adapter tensor suffix: {name}")
                continue
            pair = pairs.setdefault(base_name, {})
            if side in pair:
                report.errors.append(f"duplicate LoRA {side.upper()} tensor for {base_name}")
            pair[side] = info

        for base_name, pair in sorted(pairs.items()):
            if "a" not in pair or "b" not in pair:
                report.errors.append(f"incomplete LoRA tensor pair for {base_name}")
                continue
            if base_name not in base.tensors:
                report.errors.append(f"adapter target is absent from base model: {base_name}")
                continue

            base_info = base.tensors[base_name]
            a_info = pair["a"]
            b_info = pair["b"]
            if not base_info.is_matrix or not a_info.is_matrix or not b_info.is_matrix:
                report.errors.append(f"LoRA target and factors must be rank-2: {base_name}")
                continue

            rank = a_info.out_features
            report.ranks.add(rank)
            if base_info.in_features != a_info.in_features:
                report.errors.append(
                    f"A input dimension mismatch for {base_name}: "
                    f"{a_info.in_features} != {base_info.in_features}"
                )
            if base_info.out_features != b_info.out_features:
                report.errors.append(
                    f"B output dimension mismatch for {base_name}: "
                    f"{b_info.out_features} != {base_info.out_features}"
                )
            if rank != b_info.in_features:
                report.errors.append(
                    f"LoRA rank mismatch for {base_name}: "
                    f"A rank {rank} != B rank {b_info.in_features}"
                )
            report.tensor_pairs += 1

        if len(report.ranks) > 1:
            report.warnings.append(
                "adapter contains multiple ranks; adapter.lora.alpha is global"
            )
        alpha = adapter.metadata.get("adapter.lora.alpha")
        if not isinstance(alpha, (int, float)):
            report.errors.append("adapter.lora.alpha must be numeric")
        elif len(report.ranks) == 1 and float(alpha) != float(next(iter(report.ranks))):
            report.warnings.append(
                f"alpha/rank runtime multiplier is {float(alpha) / next(iter(report.ranks)):.6g}, not 1"
            )

    if report.tensor_pairs == 0:
        report.errors.append("adapter contains no complete LoRA tensor pairs")
    return report
