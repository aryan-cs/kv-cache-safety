from cache_safety_erasure.models.loader import _device_map_has_cpu_or_disk


def test_device_map_offload_guard_detects_cpu_and_disk() -> None:
    assert _device_map_has_cpu_or_disk({"embed": 0, "lm_head": "cpu"}) is True
    assert _device_map_has_cpu_or_disk({"embed": 0, "block": "disk"}) is True
    assert _device_map_has_cpu_or_disk({"embed": 0, "block": "cuda:0"}) is False
