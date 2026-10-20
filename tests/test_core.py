"""核心单元测试：帧结构、差分、动作解析、编码缓存。

运行：
    py -3.12 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_retina.brain import parse_action  # noqa: E402
from vision_retina.bus import Observation, grid_diff  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.frames import Rect, RetinaFrame  # noqa: E402
from vision_retina.service import RetinaService, _frame_signature  # noqa: E402


def make_frame(frame_id: int = 1, size: tuple[int, int] = (64, 48), fill: int = 0) -> RetinaFrame:
    width, height = size
    pixels = np.full((height, width, 3), fill, dtype=np.uint8)
    return RetinaFrame(
        frame_id=frame_id,
        timestamp_ms=int(time.time() * 1000),
        source="desktop",
        pixels=pixels,
    )


class FrameTests(unittest.TestCase):
    def test_rect_iteration_and_dimensions(self) -> None:
        rect = Rect(10, 20, 30, 40)
        self.assertEqual(tuple(rect), (10, 20, 30, 40))
        self.assertEqual(len(rect), 4)
        self.assertEqual(rect[2], 30)
        self.assertEqual(rect.area, 1200)
        self.assertEqual(rect.right, 40)
        self.assertEqual(rect.bottom, 60)

    def test_frame_signature_changes_with_content(self) -> None:
        self.assertEqual(_frame_signature(make_frame(fill=0)), _frame_signature(make_frame(fill=0)))
        self.assertNotEqual(_frame_signature(make_frame(fill=0)), _frame_signature(make_frame(fill=9)))


class GridDiffTests(unittest.TestCase):
    def test_identical_frames_have_no_dirty_rects(self) -> None:
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        rects, ratio = grid_diff(frame, frame.copy(), grid=32)
        self.assertEqual(rects, [])
        self.assertEqual(ratio, 0.0)

    def test_changed_block_produces_dirty_rect(self) -> None:
        before = np.zeros((128, 128, 3), dtype=np.uint8)
        after = before.copy()
        after[32:64, 32:64] = 255
        rects, ratio = grid_diff(before, after, grid=32)
        self.assertTrue(rects)
        self.assertGreater(ratio, 0.0)
        union = Rect(0, 0, 0, 0)
        for rect in rects:
            union = union.union(rect)
        self.assertTrue(union.intersects(Rect(32, 32, 32, 32)))


class ParseActionTests(unittest.TestCase):
    def test_nested_target_object(self) -> None:
        action = parse_action(
            '{"action":"click","target":{"x":670,"y":527},"params":{},"meta":{"frame_id":3}}'
        )
        self.assertIsNotNone(action)
        self.assertEqual(action["action"], "click")
        self.assertEqual(action["target"]["x"], 670)

    def test_fenced_json(self) -> None:
        action = parse_action('```json\n{"action":"type","params":{"text":"demo"}}\n```')
        self.assertIsNotNone(action)
        self.assertEqual(action["params"]["text"], "demo")

    def test_rejects_unknown_action(self) -> None:
        self.assertIsNone(parse_action('{"action":"delete_everything"}'))
        self.assertIsNone(parse_action("no action here"))


class RegionWhitelistTests(unittest.TestCase):
    def test_whitelist_accepts_inside_and_rejects_outside(self) -> None:
        config = load_config()
        config.data.setdefault("safety", {})["region_whitelist"] = [[0, 0, 800, 600]]
        service = RetinaService(config)
        inside = service.set_region((100, 100, 400, 300))
        self.assertEqual(inside["region"], [100, 100, 400, 300])
        with self.assertRaises(ValueError):
            service.set_region((700, 0, 400, 300))
        cleared = service.set_region(None)
        self.assertEqual(cleared["region"], [0, 0, 800, 600])

    def test_safety_status_reports_switch_defaults(self) -> None:
        service = RetinaService(load_config())
        safety = service.safety_status()
        self.assertFalse(safety["persist_frames"])
        self.assertFalse(safety["allow_record"])


class EncodeCacheTests(unittest.TestCase):
    def test_same_pixels_are_not_reencoded(self) -> None:
        config = load_config()
        service = RetinaService(config)
        frame = make_frame(frame_id=1)

        class FakeBus:
            def __init__(self) -> None:
                self.calls = 0

            def observe(self, force_keyframe: bool = False) -> Observation:
                self.calls += 1
                # 新 frame_id、相同像素：模拟静止画面上的重复帧
                current = RetinaFrame(
                    frame_id=self.calls,
                    timestamp_ms=int(time.time() * 1000),
                    source="desktop",
                    pixels=frame.pixels,
                    is_key_frame=self.calls == 1,
                )
                return Observation(
                    kind="frame",
                    seq=self.calls,
                    source="desktop",
                    frame=current,
                    raw_size=(current.width, current.height),
                )

            def close(self) -> None:
                pass

        service._bus = FakeBus()  # type: ignore[assignment]
        service._started_at = time.monotonic()

        first = service.observe(include_image=True)
        second = service.observe(include_image=True)
        self.assertIsNotNone(first.image)
        self.assertEqual(first.image, second.image)
        self.assertEqual(service._counters["encode"], 1)
        self.assertEqual(service._counters["encode_hits"], 1)
        service._bus = None


if __name__ == "__main__":
    unittest.main()
