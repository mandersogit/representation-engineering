from __future__ import annotations

from pathlib import Path

import torch

from repreng_gguf.adapter import ProjectionTarget, compile_projection_adapter
from repreng_gguf.gguf_reader import GGUFReader
from repreng_gguf.gguf_writer import GGUFWriter
from repreng_gguf.projection import explicit_project_weight
from repreng_gguf.quantization import GGMLQuantizationType, dequantize_rows


def read_dense_tensor(path: Path, name: str) -> torch.Tensor:
    with GGUFReader(path) as reader:
        info = reader.tensor_info(name)
        raw = list(reader.iter_raw_rows(name, rows_per_chunk=info.out_features))
        assert len(raw) == 1
        values = dequantize_rows(raw[0].data, info.quantization, info.in_features)
        return torch.from_numpy(values.copy())


def make_base_model(path: Path) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260722)
    weight = torch.randn(7, 32, generator=generator, dtype=torch.float32)
    writer = GGUFWriter()
    writer.add_string("general.type", "model")
    writer.add_string("general.architecture", "qwen2")
    writer.add_string("general.name", "adapter writer fixture")
    writer.add_tensor(
        "blk.0.attn_output.weight",
        weight.numpy(),
        quantization=GGMLQuantizationType.F16,
    )
    writer.write(path)
    return weight.to(torch.float16).to(torch.float32)


def test_compile_projection_adapter_writes_llama_cpp_layout(tmp_path: Path) -> None:
    base_path = tmp_path / "base.gguf"
    adapter_path = tmp_path / "projection.gguf"
    quantized_weight = make_base_model(base_path)
    generator = torch.Generator().manual_seed(99)
    directions = torch.randn(7, 2, generator=generator)

    factors = compile_projection_adapter(
        base_path,
        adapter_path,
        [
            ProjectionTarget(
                "blk.0.attn_output.weight",
                directions,
                strength=0.75,
            )
        ],
        max_chunk_bytes=64,
        tensor_type=GGMLQuantizationType.F16,
    )

    with GGUFReader(adapter_path) as reader:
        assert reader.metadata["general.type"] == "adapter"
        assert reader.metadata["general.architecture"] == "qwen2"
        assert reader.metadata["adapter.type"] == "lora"
        assert reader.metadata["adapter.lora.alpha"] == 2.0

        a_info = reader.tensor_info("blk.0.attn_output.weight.lora_a")
        b_info = reader.tensor_info("blk.0.attn_output.weight.lora_b")
        assert a_info.logical_shape == (2, 32)
        assert b_info.logical_shape == (7, 2)
        assert a_info.quantization == GGMLQuantizationType.F16
        assert b_info.quantization == GGMLQuantizationType.F16

        with GGUFReader(base_path) as base_reader:
            base_info = base_reader.tensor_info("blk.0.attn_output.weight")
            assert base_info.in_features == a_info.in_features
            assert base_info.out_features == b_info.out_features
            assert a_info.out_features == b_info.in_features

    a = read_dense_tensor(adapter_path, "blk.0.attn_output.weight.lora_a")
    b = read_dense_tensor(adapter_path, "blk.0.attn_output.weight.lora_b")
    actual = quantized_weight + b @ a
    expected = explicit_project_weight(quantized_weight, directions, strength=0.75)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    original = factors["blk.0.attn_output.weight"]
    torch.testing.assert_close(a, original.A.to(torch.float16).to(torch.float32))
    torch.testing.assert_close(b, original.B.to(torch.float16).to(torch.float32))


def test_adapter_rejects_mixed_ranks(tmp_path: Path) -> None:
    from repreng_gguf.adapter import write_lora_adapter
    from repreng_gguf.projection import LoRAFactors

    factors = {
        "one.weight": LoRAFactors(A=torch.zeros(1, 4), B=torch.zeros(3, 1)),
        "two.weight": LoRAFactors(A=torch.zeros(2, 4), B=torch.zeros(3, 2)),
    }
    try:
        write_lora_adapter(tmp_path / "bad.gguf", architecture="qwen2", factors=factors)
    except ValueError as error:
        assert "same rank" in str(error)
    else:
        raise AssertionError("mixed-rank adapter was accepted")
