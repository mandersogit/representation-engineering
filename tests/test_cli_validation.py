from __future__ import annotations

from pathlib import Path

from safetensors.torch import save_file
import torch

from repreng_gguf.adapter import write_lora_adapter
from repreng_gguf.cli import main
from repreng_gguf.gguf_writer import GGUFWriter
from repreng_gguf.projection import LoRAFactors
from repreng_gguf.quantization import GGMLQuantizationType
from repreng_gguf.validation import validate_lora_adapter


TENSOR_NAME = "blk.0.attn_output.weight"


def write_base(path: Path) -> torch.Tensor:
    generator = torch.Generator().manual_seed(700)
    weight = torch.randn(9, 256, generator=generator)
    writer = GGUFWriter()
    writer.add_string("general.type", "model")
    writer.add_string("general.architecture", "qwen2")
    writer.add_tensor(TENSOR_NAME, weight.numpy(), quantization=GGMLQuantizationType.F16)
    writer.write(path)
    return weight


def test_cli_inspect_compile_and_validate(tmp_path: Path, capsys) -> None:
    base = tmp_path / "base.gguf"
    directions_path = tmp_path / "directions.safetensors"
    adapter = tmp_path / "adapter.gguf"
    write_base(base)
    directions = torch.randn(9, 3, generator=torch.Generator().manual_seed(701))
    save_file({TENSOR_NAME: directions}, directions_path)

    assert main(["inspect", str(base), "--supported-only"]) == 0
    inspect_output = capsys.readouterr().out
    assert TENSOR_NAME in inspect_output
    assert "F16" in inspect_output

    assert (
        main(
            [
                "compile",
                str(base),
                str(directions_path),
                str(adapter),
                "--chunk-mib",
                "0.001",
                "--strength",
                "0.5",
            ]
        )
        == 0
    )
    compile_output = capsys.readouterr().out
    assert "1 tensor pairs" in compile_output
    assert adapter.exists()

    assert main(["validate", str(base), str(adapter), "--json"]) == 0
    validation_output = capsys.readouterr().out
    assert '"valid": true' in validation_output
    assert '"ranks": [' in validation_output


def test_validator_reports_architecture_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    adapter = tmp_path / "adapter.gguf"
    write_base(base)
    factors = {
        TENSOR_NAME: LoRAFactors(A=torch.zeros(2, 256), B=torch.zeros(9, 2))
    }
    write_lora_adapter(adapter, architecture="llama", factors=factors)

    report = validate_lora_adapter(base, adapter)
    assert not report.valid
    assert any("architecture mismatch" in error for error in report.errors)
