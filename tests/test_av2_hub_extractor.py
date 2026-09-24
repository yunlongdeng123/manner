import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "extract_av2_hub_images.py"
SPEC = importlib.util.spec_from_file_location("extract_av2_hub_images", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_member_camera_key():
    name = "sensor/val/log-id/sensors/cameras/ring_front_center/123.jpg"
    assert MODULE.member_camera_key(name) == ("log-id", "ring_front_center", "123.jpg")
    assert MODULE.member_camera_key("sensor/val/log-id/city_SE3_egovehicle.feather") is None


def test_safe_target_rejects_traversal(tmp_path):
    target = MODULE.safe_target(tmp_path, "val/segment/image/cam/1.jpg")
    assert tmp_path.resolve() in target.parents
    try:
        MODULE.safe_target(tmp_path, "../escape.jpg")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal was not rejected")


def test_build_requests_from_compact_index(tmp_path):
    index = tmp_path / "requests.jsonl"
    index.write_text(
        '{"frame_key":"val/1/2","source_split":"val","source_id":"log","camera":"ring_front_center",'
        '"filename":"2.jpg","image_path":"val/1/image/ring_front_center/2.jpg"}\n',
        encoding="utf-8",
    )
    requests, report = MODULE.build_requests_from_index(tmp_path, index, ["val"])
    assert ("log", "ring_front_center", "2.jpg") in requests
    assert report == {"frames": 1, "images_requested": 1, "unique_source_images": 1}
