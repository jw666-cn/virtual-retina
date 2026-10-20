"""服务层：静默帧返回缓存图片（不重新编码），且元数据正确标记。"""
from __future__ import annotations

import time
import unittest

import numpy as np

from vision_retina.bus import Observation
from vision_retina.config import load_config
from vision_retina.frames import RetinaFrame
from vision_retina.service import RetinaService


def _frame(frame_id: int = 1) -> RetinaFrame:
    return RetinaFrame(
        frame_id=frame_id,
        timestamp_ms=int(time.time() * 1000),
        source="desktop",
        pixels=np.zeros((40, 60, 3), dtype=np.uint8),
        dirty_rects=[],
    )


class _FakeBus:
    def __init__(self, observations: list[Observation]):
        self._queue = list(observations)
        self.region = None
        self.capture = type("FakeCapture", (), {"describe": lambda self: {"class": "FakeCapture"}})()

    def start(self):
        return self

    def close(self):
        pass

    def observe(self, force_keyframe: bool = False) -> Observation:
        return self._queue.pop(0)

    def set_region(self, region):
        self.region = region

    def status(self):
        return {"mode": "delta"}


class CachedObserveTests(unittest.TestCase):
    def setUp(self):
        config = load_config()
        config.data["encode"]["max_width"] = 0
        self.service = RetinaService(config)
        self.service._bus = _FakeBus([
            Observation(kind="frame", seq=1, source="desktop", frame=_frame(1), raw_size=(60, 40)),
            Observation(kind="unchanged", seq=2, source="desktop", frame=None, raw_size=(60, 40), reason="无变化"),
        ])

    def tearDown(self):
        self.service.close()

    def test_unchanged_observe_reuses_cached_image(self):
        first = self.service.observe(include_image=True)
        second = self.service.observe(include_image=True)

        self.assertFalse(first.cached_image)
        self.assertTrue(first.has_image)
        self.assertEqual(first.image[:2], b"\xff\xd8")  # JPEG SOI

        self.assertEqual(second.observation.kind, "unchanged")
        self.assertTrue(second.has_image)
        self.assertTrue(second.cached_image)
        self.assertEqual(second.image, first.image)
        self.assertTrue(second.meta()["image"]["cached"])


if __name__ == "__main__":
    unittest.main()
