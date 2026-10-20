"""仿生双路 + 指令式变焦 + token 预算的单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_retina.bionic import (  # noqa: E402
    BionicConfig, BionicRetina, estimate_image_tokens, parse_zoom,
)
from vision_retina.brain import BrainReply  # noqa: E402
from vision_retina.frames import Rect, RetinaFrame, now_ms  # noqa: E402


def make_text_frame(width: int = 640, height: int = 360) -> RetinaFrame:
    """深色底 + 一块带“文字”纹理的小区域，用于验证语义扩展。"""
    pixels = np.full((height, width, 3), 20, dtype=np.uint8)
    pixels[28:120, 40:250] = 245
    for y in range(36, 112, 12):
        pixels[y:y + 4, 48:242] = 15
    return RetinaFrame(
        frame_id=1, timestamp_ms=now_ms(), source="desktop", pixels=pixels,
        dirty_rects=[Rect(40, 28, 210, 92)], is_key_frame=True,
    )


class ScriptedBrain:
    """按脚本返回 zoom / 答案，并记录每次请求的图片数量。"""

    def __init__(self, first_text: str, second_text: str = "3.5 kW"):
        self.first_text = first_text
        self.second_text = second_text
        self.calls: list[list[bytes]] = []

    def ask_images(self, images, prompt, **kwargs) -> BrainReply:
        batch = list(images)
        self.calls.append(batch)
        text = self.first_text if len(self.calls) == 1 else self.second_text
        return BrainReply(
            text=text, model="fake", latency_ms=5,
            usage={"prompt_tokens": 320 if len(self.calls) == 1 else 720,
                   "completion_tokens": 40},
        )


class ZoomParsingTests(unittest.TestCase):
    def test_parse_nested_zoom(self) -> None:
        text = '前置说明 {"action":"zoom","region":{"x":10,"y":20,"w":300,"h":120},"reason":"小字"} 后缀'
        req = parse_zoom(text)
        self.assertIsNotNone(req)
        assert req is not None
        self.assertEqual((req.rect.x, req.rect.y, req.rect.w, req.rect.h), (10, 20, 300, 120))

    def test_parse_code_fence_and_rect_list(self) -> None:
        text = '```json\n{"action":"zoom","region":[5,6,70,80]}\n```'
        req = parse_zoom(text)
        self.assertIsNotNone(req)
        assert req is not None
        self.assertEqual(tuple(req.rect), (5, 6, 70, 80))

    def test_ignore_non_zoom_action(self) -> None:
        self.assertIsNone(parse_zoom('{"action":"click","target":{"x":1,"y":2}}'))
        self.assertIsNone(parse_zoom("看不清"))


class BionicFlowTests(unittest.TestCase):
    def test_dual_path_sends_low_then_focus(self) -> None:
        brain = ScriptedBrain(
            '{"action":"zoom","region":{"x":45,"y":30,"w":200,"h":90},"reason":"读功率"}'
        )
        frame = make_text_frame()
        engine = BionicRetina(brain, bionic_config=BionicConfig(budget_tokens=1400, low_width=480))
        result = engine.ask("输入功率是多少？", frame)
        self.assertEqual(result.calls, 2)
        self.assertEqual([len(call) for call in brain.calls], [1, 2])
        self.assertIsNotNone(result.focus)
        assert result.focus is not None and result.zoom is not None
        # 语义扩展：请求框未覆盖整块文字时，应扩到整块内容
        self.assertLessEqual(result.zoom.x, 45)
        self.assertLessEqual(result.zoom.y, 30)
        self.assertGreaterEqual(result.zoom.right, 245)
        self.assertGreaterEqual(result.zoom.bottom, 120)
        self.assertTrue(result.budget_ok)
        self.assertEqual(result.answer, "3.5 kW")
        self.assertEqual(result.low.width, 480)

    def test_budget_denies_zoom(self) -> None:
        brain = ScriptedBrain('{"action":"zoom","region":{"x":0,"y":0,"w":640,"h":360}}')
        frame = make_text_frame()
        engine = BionicRetina(brain, bionic_config=BionicConfig(budget_tokens=120, low_width=640))
        result = engine.ask("读整页", frame)
        self.assertEqual(result.calls, 1)
        self.assertIsNone(result.focus)
        self.assertFalse(result.budget_ok)
        self.assertIsNotNone(result.denied_reason)

    def test_no_zoom_when_low_view_enough(self) -> None:
        brain = ScriptedBrain("这是减速器装配图。")
        frame = make_text_frame()
        engine = BionicRetina(brain, bionic_config=BionicConfig(budget_tokens=900))
        result = engine.ask("这是什么图？", frame)
        self.assertEqual(result.calls, 1)
        self.assertEqual(len(brain.calls), 1)
        self.assertEqual(result.answer, "这是减速器装配图。")

    def test_token_estimate_is_monotonic(self) -> None:
        self.assertLess(estimate_image_tokens(320, 240), estimate_image_tokens(1280, 960))
        self.assertGreater(estimate_image_tokens(640, 480), 0)


if __name__ == "__main__":
    unittest.main()
