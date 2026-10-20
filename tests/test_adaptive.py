"""场景自适应编码（仿暗适应）与总线联动测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_retina.adaptive import (  # noqa: E402
    SCENE_DYNAMIC, SCENE_NORMAL, SCENE_STATIC, AdaptiveController,
)
from vision_retina.bus import RetinaBus  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.frames import Rect, RetinaFrame, now_ms  # noqa: E402


class FakeCapture:
    source = "desktop"
    region = None
    fps = 2.0

    def stats(self) -> dict:
        return {"captured": 0, "errors": 0}

    def close(self) -> None:
        return None


def make_frame(value: int = 0, frame_id: int = 1) -> RetinaFrame:
    pixels = np.full((180, 320, 3), value, dtype=np.uint8)
    return RetinaFrame(
        frame_id=frame_id, timestamp_ms=now_ms(), source="desktop", pixels=pixels,
        dirty_rects=[Rect(0, 0, 320, 180)], is_key_frame=True,
    )


class AdaptiveControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctl = AdaptiveController(load_config())

    def test_enters_static_after_dwell(self) -> None:
        for _ in range(2):
            self.ctl.update(1.0, 0.0)
        self.assertEqual(self.ctl.current.name, SCENE_NORMAL)
        self.ctl.update(1.0, 0.0)
        self.assertEqual(self.ctl.current.name, SCENE_STATIC)
        self.assertLess(self.ctl.current.fps, 2.0)
        self.assertLess(self.ctl.current.quality, 80)

    def test_enters_dynamic_quickly(self) -> None:
        self.ctl.update(0.5, 0.5)
        self.assertEqual(self.ctl.current.name, SCENE_DYNAMIC)
        self.assertGreater(self.ctl.current.fps, 2.0)
        self.assertGreater(self.ctl.current.quality, 80)

    def test_hysteresis_holds_static_on_small_change(self) -> None:
        for _ in range(4):
            self.ctl.update(1.0, 0.0)
        self.assertEqual(self.ctl.current.name, SCENE_STATIC)
        # 少量变化（0.004）低于退出阈值 0.006，应保持静态档
        self.ctl.update(1.0, 0.004)
        self.assertEqual(self.ctl.current.name, SCENE_STATIC)

    def test_status_shape(self) -> None:
        self.ctl.update(0.2, 0.01)
        status = self.ctl.status()
        self.assertIn("scene", status)
        self.assertIn("ema_change_ratio", status)

    def test_disabled_always_normal(self) -> None:
        cfg = load_config()
        cfg.data["adaptive"]["enabled"] = False
        ctl = AdaptiveController(cfg)
        for _ in range(5):
            ctl.update(1.0, 0.0)
        self.assertEqual(ctl.current.name, SCENE_NORMAL)


class BusAdaptiveTests(unittest.TestCase):
    def test_bus_attaches_scene_profile(self) -> None:
        clock = {"t": 0.0}

        def now() -> float:
            return clock["t"]

        cfg = load_config()
        cfg.data["bus"]["mode"] = "delta"
        bus = RetinaBus(FakeCapture(), cfg, clock=now)  # type: ignore[arg-type]
        frame = make_frame(0)
        first = bus.evaluate(frame, force_keyframe=True)
        self.assertEqual(first.kind, "frame")
        self.assertIsNotNone(first.frame)
        assert first.frame is not None
        self.assertIsNotNone(first.frame.meta.get("scene"))

        for _ in range(4):
            clock["t"] += 1.0
            bus.evaluate(frame)
        self.assertIsNotNone(bus.scene_profile)
        status = bus.status()
        self.assertIn("adaptive", status)
        self.assertIn(status["adaptive"]["scene"]["name"], (SCENE_STATIC, SCENE_NORMAL, SCENE_DYNAMIC))


if __name__ == "__main__":
    unittest.main()
