import pytest
import torch

from tinydiffusion.utils.device import bf16_supported, describe_device, resolve_device


def test_none_follows_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(None) == expected


def test_cpu_is_left_alone():
    assert resolve_device("cpu") == "cpu"


def test_an_unknown_device_is_a_user_error():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("gpu")


def test_cuda_falls_back_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:0") == "cpu"
    assert "falling back to CPU" in capsys.readouterr().out


def test_fallback_can_be_quiet(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda", say=None) == "cpu"
    assert capsys.readouterr().out == ""


def test_describe_device_is_plain_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert describe_device("cpu") == "cpu"
    assert describe_device("cuda") == "cuda"


def test_describe_device_names_the_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5060")
    assert describe_device("cuda") == "cuda (NVIDIA GeForce RTX 5060)"


def test_emulated_bfloat16_does_not_count_as_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: including_emulation,
    )

    assert not bf16_supported()


def test_native_bfloat16_counts_as_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)

    assert bf16_supported()


def test_bfloat16_is_not_supported_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert not bf16_supported()


def test_older_torch_without_the_emulation_keyword_still_answers(monkeypatch):
    def old_signature():
        return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", old_signature)

    assert bf16_supported()
