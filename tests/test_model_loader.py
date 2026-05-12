import sys
from types import ModuleType, SimpleNamespace

from cache_safety_erasure.config import ModelConfig
from cache_safety_erasure.models.loader import _device_map_has_cpu_or_disk, _load_hf_model


def test_device_map_offload_guard_detects_cpu_and_disk() -> None:
    assert _device_map_has_cpu_or_disk({"embed": 0, "lm_head": "cpu"}) is True
    assert _device_map_has_cpu_or_disk({"embed": 0, "block": "disk"}) is True
    assert _device_map_has_cpu_or_disk({"embed": 0, "block": "cuda:0"}) is False


def test_hf_loader_applies_peft_adapter_and_tokenizer_source(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class FakeModel:
        hf_device_map = {"": "cuda:0"}

        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            calls.append(("tokenizer", args, kwargs))
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            calls.append(("base_model", args, kwargs))
            return FakeModel()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: FakeModel, *args: object, **kwargs: object) -> FakeModel:
            calls.append(("adapter", (model, *args), kwargs))
            return model

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel
    torch = SimpleNamespace(float32="float32", float16="float16", bfloat16="bfloat16")
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "torch", torch)

    bundle = _load_hf_model(
        ModelConfig(
            provider="hf",
            model_id="Qwen/Qwen2.5-14B-Instruct",
            revision="base-rev",
            tokenizer_id="chloeli/qwen-2.5-14b-rules-spec-msm",
            tokenizer_revision="tok-rev",
            adapter_id="chloeli/qwen-2.5-14b-rules-spec-msm",
            adapter_revision="adapter-rev",
        )
    )

    assert bundle.model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert bundle.model.eval_called is True
    assert calls[0] == (
        "tokenizer",
        ("chloeli/qwen-2.5-14b-rules-spec-msm",),
        {
            "revision": "tok-rev",
            "trust_remote_code": False,
            "local_files_only": False,
        },
    )
    assert calls[1][0] == "base_model"
    assert calls[1][1] == ("Qwen/Qwen2.5-14B-Instruct",)
    assert calls[1][2]["revision"] == "base-rev"
    assert calls[2][0] == "adapter"
    assert calls[2][1][1:] == ("chloeli/qwen-2.5-14b-rules-spec-msm",)
    assert calls[2][2]["revision"] == "adapter-rev"
    assert calls[2][2]["is_trainable"] is False
