"""视神经转化引擎（OpticNerve）单元测试。

覆盖四类结果（markdown / image / structured / frame）、converter 注册与优先级、
视觉稀疏编码与坐标反解、流式去重、以及文档/结构化/代码/视频转换。
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_retina import optic_nerve as on
from vision_retina.bus import Observation
from vision_retina.frames import Rect, RetinaFrame


# --------------------------------------------------------------------------- #
# 合成画面
# --------------------------------------------------------------------------- #


def text_page(width: int = 1000, height: int = 700, lines: int = 22) -> np.ndarray:
    """白底黑字的模拟文档页（行内还有一条"小字"细线）。"""

    pixels = np.full((height, width, 3), 250, np.uint8)
    y = 28
    for index in range(lines):
        length = int(width * (0.55 if index % 3 else 0.35))
        pixels[y : y + 6, 46 : 46 + length] = 30
        pixels[y + 9 : y + 12, 46 : 46 + int(length * 0.7)] = 70
        y += 24
        if y > height - 30:
            break
    return pixels


def dialog_frame(width: int = 1000, height: int = 700) -> np.ndarray:
    """灰底上一个居中白色对话框（稀疏布局的典型场景）。"""

    pixels = np.full((height, width, 3), 116, np.uint8)
    pixels[180:480, 260:760] = 248
    pixels[220:228, 300:700] = 40
    pixels[250:258, 300:640] = 60
    pixels[300:340, 300:470] = 90
    pixels[300:340, 500:720] = 90
    pixels[420:452, 620:700] = 150
    return pixels


def photo_frame(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """高频"照片"：没有明显版面结构，应走 full 策略。"""

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 90, size=(height, width, 3), dtype=np.uint8)
    gradient = np.linspace(40, 210, width, dtype=np.uint8)[None, :, None]
    return np.clip(noise.astype(np.int16) + gradient.astype(np.int16), 0, 255).astype(np.uint8)


def make_frame(pixels: np.ndarray, *, frame_id: int = 1, timestamp_ms: int = 0, **kwargs) -> RetinaFrame:
    return RetinaFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source=kwargs.pop("source", "desktop"),
        pixels=pixels,
        **kwargs,
    )


def jpeg_bytes(pixels: np.ndarray) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", pixels)
    assert ok
    return buffer.tobytes()


def docx_bytes() -> bytes:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Doc Title</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>"
        "<w:tbl>"
        "<w:tr><w:tc><w:p><w:r><w:t>H1</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>H2</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


class FakeVideoReader:
    """最小视频读取器：按 (timestamp, frame) 迭代。"""

    def __init__(self, frames):
        self.frames = list(frames)
        self.meta = {"fps": 10.0, "frame_count": len(self.frames), "duration_s": len(self.frames) / 10.0}

    def __iter__(self):
        for index, pixels in enumerate(self.frames):
            yield index / 10.0, pixels


# --------------------------------------------------------------------------- #
# 结果类型
# --------------------------------------------------------------------------- #


class ResultTypeTest(unittest.TestCase):
    def test_markdown_result_keeps_text_content_alias(self):
        result = on.DocumentConverterResult(
            kind=on.ResultKind.MARKDOWN, markdown="# 标题\n\n正文", title="t"
        )
        self.assertEqual(str(result), "# 标题\n\n正文")
        self.assertEqual(result.text_content, result.markdown)
        self.assertEqual(result.image_bytes(), [])
        self.assertEqual(result.all_frames(), [])

    def test_visual_result_has_empty_markdown_and_openai_parts(self):
        payload = jpeg_bytes(dialog_frame(320, 200))
        frame = on.EncodedFrame(data=payload, width=320, height=200, tokens=120)
        result = on.DocumentConverterResult(
            kind=on.ResultKind.IMAGE, markdown="", images=[frame], tokens=on.TokenReport(400, 120)
        )
        self.assertEqual(result.text_content, "")
        self.assertEqual(result.image_bytes(), [payload])
        parts = result.to_openai_content("这是什么")
        self.assertEqual(parts[0], {"type": "text", "text": "这是什么"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIs(result.all_frames()[0], frame)

    def test_encoded_frame_roundtrip(self):
        payload = jpeg_bytes(dialog_frame(200, 120))
        frame = on.EncodedFrame(data=payload, width=200, height=120, tokens=90, strategy="sparse")
        restored = on.decode_image(payload)
        self.assertEqual(restored.shape[:2], (120, 200))
        self.assertEqual(frame.to_dict()["tokens"], 90)
        self.assertTrue(frame.to_data_uri().startswith("data:image/jpeg;base64,"))
        self.assertTrue(frame.to_openai_part()["image_url"]["url"].startswith("data:image/jpeg"))

    def test_structured_payload_markdown_and_compact(self):
        payload = on.StructuredPayload(
            format="table",
            columns=["名称", "数量"],
            rows=[["螺栓", "12"], ["垫片", "8"]],
            records=[{"名称": "螺栓", "数量": "12"}, {"名称": "垫片", "数量": "8"}],
        )
        markdown = payload.to_markdown()
        self.assertIn("| 名称 | 数量 |", markdown)
        self.assertIn("| 螺栓 | 12 |", markdown)
        self.assertIn("螺栓\t12", payload.to_compact())
        self.assertEqual(payload.shape, (2, 2))
        self.assertEqual(len(payload.to_records()), 2)

    def test_token_report_ratio(self):
        report = on.TokenReport(original=400, output=120, strategy="sparse")
        self.assertAlmostEqual(report.ratio, 0.3, places=4)
        self.assertEqual(report.saved, 280)
        self.assertAlmostEqual(report.saved_ratio, 0.7, places=4)
        self.assertEqual(report.to_dict()["output"], 120)

    def test_result_to_dict_can_omit_payload(self):
        payload = jpeg_bytes(dialog_frame(160, 100))
        frame = on.EncodedFrame(data=payload, width=160, height=100, tokens=60)
        result = on.DocumentConverterResult(kind=on.ResultKind.IMAGE, image=frame, tokens=on.TokenReport(200, 60))
        data = result.to_dict(include_payload=False)
        self.assertEqual(data["kind"], "image")
        self.assertIsInstance(data["image"]["bytes"], int)
        self.assertNotIn("data", data["image"])
        self.assertIn("tokens", data)


# --------------------------------------------------------------------------- #
# 注册表 / 引擎
# --------------------------------------------------------------------------- #


class _UppercaseConverter(on.DocumentConverter):
    name = "upper"

    def __init__(self):
        self.positions = []

    def accepts(self, file_stream, stream_info, **kwargs):  # noqa: D401
        self.positions.append(file_stream.tell())
        return ".upper" in (stream_info.extension or "")

    def convert(self, file_stream, stream_info, **kwargs):
        self.positions.append(file_stream.tell())
        return on.DocumentConverterResult(
            kind=on.ResultKind.MARKDOWN, markdown=file_stream.read().decode("utf-8").upper()
        )


class _BoomConverter(on.DocumentConverter):
    name = "boom"

    def accepts(self, file_stream, stream_info, **kwargs):
        return ".boom" in (stream_info.extension or "")

    def convert(self, file_stream, stream_info, **kwargs):
        raise on.MissingDependencyException("需要 boom 库")


class _FallbackProbe(on.DocumentConverter):
    name = "probe"
    priority = 5.0

    def accepts(self, file_stream, stream_info, **kwargs):
        return True

    def convert(self, file_stream, stream_info, **kwargs):
        return on.DocumentConverterResult(kind=on.ResultKind.MARKDOWN, markdown="probe")


class RegistryTest(unittest.TestCase):
    def test_document_converter_is_abstract(self):
        with self.assertRaises(TypeError):
            on.DocumentConverter()  # type: ignore[abstract]

    def test_custom_converter_is_used_and_stream_position_reset(self):
        converter = _UppercaseConverter()
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(converter, priority=0.0)
        result = engine.convert(b"hello", stream_info=on.StreamInfo(extension=".upper", mimetype="text/plain"))
        self.assertEqual(result.markdown, "HELLO")
        self.assertIn(0, converter.positions)

    def test_same_type_registration_replaces_previous(self):
        first = _UppercaseConverter()
        second = _UppercaseConverter()
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(first, priority=9.0)
        engine.register_converter(second, priority=1.0)
        # 同类只保留最新一次注册（插件可热替换）
        names = [item["name"] for item in engine.status()["converters"]]
        self.assertEqual(names, ["upper"])

    def test_priority_orders_custom_converters(self):
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(_UppercaseConverter(), priority=50.0)
        engine.register_converter(_FallbackProbe(), priority=0.0)
        result = engine.convert(b"x", stream_info=on.StreamInfo(extension=".upper", mimetype="text/plain"))
        self.assertEqual(result.markdown, "probe")

    def test_missing_dependency_propagates(self):
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(_BoomConverter(), priority=0.0)
        with self.assertRaises(on.MissingDependencyException):
            engine.convert(b"x", stream_info=on.StreamInfo(extension=".boom"))

    def test_unknown_binary_raises_unsupported(self):
        engine = on.OpticNerve()
        with self.assertRaises(on.UnsupportedContentError):
            engine.convert(b"\x00\x01\x02\x03\xff\xfe\x00\x11")

    def test_status_lists_registry_and_counters(self):
        engine = on.OpticNerve()
        status = engine.status()
        self.assertEqual(status["name"], "optic_nerve")
        self.assertGreaterEqual(len(status["converters"]), 10)
        self.assertIn("stats", status)
        self.assertEqual(status["errors"], [])

    def test_sniff_guesses_image_and_text(self):
        engine = on.OpticNerve()
        png = jpeg_bytes(dialog_frame(80, 60))
        guesses = engine.sniff(io.BytesIO(png))
        self.assertTrue(any(item.mimetype == "image/png" for item in guesses))
        text_guesses = engine.sniff(io.BytesIO(b"plain words here\n"))
        self.assertTrue(any(item.mimetype == "text/plain" for item in text_guesses))

    def test_converter_error_is_recorded_and_next_converter_tried(self):
        class Broken(on.DocumentConverter):
            name = "broken"

            def accepts(self, file_stream, stream_info, **kwargs):
                return ".txt" in (stream_info.extension or "")

            def convert(self, file_stream, stream_info, **kwargs):
                raise RuntimeError("boom")

        engine = on.OpticNerve()
        engine.register_converter(Broken(), priority=0.0)
        result = engine.convert(b"hello", stream_info=on.StreamInfo(extension=".txt", mimetype="text/plain"))
        self.assertEqual(result.markdown, "hello")
        self.assertTrue(engine.status()["errors"])


# --------------------------------------------------------------------------- #
# 视觉：像素 / 稀疏编码 / 坐标
# --------------------------------------------------------------------------- #


class VisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = on.OpticNerve()

    def test_frame_tokens_drop_against_full_frame_baseline(self):
        # 稀疏版面（对话框四周大片背景）：舍弃背景、只留内容块
        result = self.engine.convert(make_frame(dialog_frame(1000, 700)), scene="ui")
        self.assertIs(result.kind, on.ResultKind.IMAGE)
        self.assertEqual(result.markdown, "")
        self.assertEqual(result.meta["mode"], "sparse")
        self.assertGreater(result.tokens.original, 0)
        self.assertLess(result.tokens.output, result.tokens.original * 0.75)
        self.assertLess(result.tokens.ratio, 0.75)

    def test_dense_text_page_never_costs_more_than_full_frame(self):
        # 密排文本页：宁可 token 持平（保可读性），也绝不比整帧更贵
        result = self.engine.convert(make_frame(text_page()))
        self.assertLessEqual(result.tokens.output, result.tokens.original * 1.02)
        self.assertEqual(len(result.all_frames()), 1)

    def test_dynamic_scene_keeps_full_frame(self):
        result = self.engine.convert(make_frame(text_page()), scene="dynamic")
        self.assertEqual(result.meta["mode"], "full")
        self.assertAlmostEqual(result.tokens.ratio, 1.0, delta=0.35)
        self.assertEqual(len(result.all_frames()), 1)

    def test_static_scene_is_cheaper_than_normal(self):
        frame = make_frame(dialog_frame(900, 620))
        normal = self.engine.convert(frame).tokens.output
        static = self.engine.convert(frame, scene="static").tokens.output
        self.assertLessEqual(static, normal)

    def test_small_text_is_upscaled(self):
        pixels = text_page(900, 620)
        result = self.engine.convert(make_frame(pixels))
        blocks = on.detect_blocks(pixels)
        self.assertTrue(blocks)
        self.assertLess(max(block.text_px for block in blocks), 16.0)
        self.assertEqual(result.meta["mode"], "sparse")
        placement = result.mapper.placements[0]
        self.assertGreaterEqual(placement.scale, 1.0)
        self.assertLessEqual(placement.scale, 2.05)

    def test_enhance_raises_readability_metrics(self):
        pixels = text_page(600, 400)
        enhanced = on.enhance_pixels(pixels)
        self.assertEqual(enhanced.shape, pixels.shape)
        before = on.readability(pixels)
        after = on.readability(enhanced)
        self.assertGreaterEqual(after["rms_contrast"], before["rms_contrast"])
        self.assertGreaterEqual(after["score"], before["score"] - 1e-6)

    def test_result_meta_reports_readability_improvement(self):
        result = self.engine.convert(make_frame(text_page(700, 480)))
        meta = result.meta
        self.assertIn("readability", meta)
        self.assertGreaterEqual(meta["readability_enhanced"]["score"], meta["readability"]["score"] - 1e-9)

    def test_mapper_roundtrip_for_each_block(self):
        pixels = dialog_frame(800, 560)
        result = self.engine.convert(make_frame(pixels))
        self.assertTrue(result.mapper.placements)
        for placement in result.mapper.placements:
            native = placement.native
            mosaic = result.mapper.to_mosaic(native)
            self.assertIsNotNone(mosaic)
            restored = result.mapper.to_native(mosaic)
            self.assertIsNotNone(restored)
            self.assertAlmostEqual(restored.x, native.x, delta=max(3, native.w * 0.06))
            self.assertAlmostEqual(restored.y, native.y, delta=max(3, native.h * 0.06))

    def test_to_native_point_clamps_outside_coordinates(self):
        result = self.engine.convert(make_frame(dialog_frame(640, 420)))
        x, y = result.mapper.to_native_point(-5, -5)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        far = result.mapper.to_native_point(10_000, 10_000)
        self.assertLessEqual(far[0], result.mapper.native_width)
        self.assertLessEqual(far[1], result.mapper.native_height)
        point = result.mapper.placements[0]
        centre = result.mapper.to_native_point(
            point.mosaic.x + max(1, point.mosaic.w // 2), point.mosaic.y + max(1, point.mosaic.h // 2)
        )
        self.assertIsNotNone(centre)

    def test_dirty_rects_boost_salience(self):
        pixels = text_page(700, 480)
        rect = Rect(46, 28, 300, 60)
        result = self.engine.convert(make_frame(pixels, dirty_rects=[rect]), scene="ui")
        self.assertEqual(result.meta["mode"], "sparse")
        self.assertGreaterEqual(result.meta["blocks_kept"], 1)

    def test_dual_mode_returns_low_and_focus_frames(self):
        result = self.engine.convert(make_frame(dialog_frame(900, 620)), mode="dual")
        frames = result.all_frames()
        self.assertEqual(len(frames), 2)
        self.assertLess(frames[0].width, frames[1].width)
        self.assertEqual(result.meta["mode"], "dual")

    def test_zoom_returns_crop_only(self):
        pixels = text_page(900, 620)
        region = Rect(46, 120, 320, 80)
        result = self.engine.zoom(make_frame(pixels), region)
        self.assertIs(result.kind, on.ResultKind.IMAGE)
        frames = result.all_frames()
        self.assertEqual(len(frames), 1)
        self.assertLess(frames[0].width, pixels.shape[1])
        self.assertLess(result.tokens.output, result.tokens.original)

    def test_photo_like_frame_uses_full_strategy(self):
        result = self.engine.convert(make_frame(photo_frame(480, 320)))
        self.assertEqual(result.meta["mode"], "full")
        self.assertEqual(len(result.all_frames()), 1)

    def test_tiny_frame_does_not_crash(self):
        small = np.full((24, 24, 3), 128, np.uint8)
        result = self.engine.convert(make_frame(small))
        self.assertIs(result.kind, on.ResultKind.IMAGE)
        self.assertGreater(len(result.all_frames()[0].data), 0)

    def test_token_budget_is_respected(self):
        engine = on.OpticNerve(options=on.VisionOptions(max_tokens=420, target_ratio=0.55))
        result = engine.convert(make_frame(text_page(1100, 760)))
        self.assertLessEqual(result.tokens.output, 420)
        self.assertEqual(result.meta["mode"], "sparse")

    def test_options_from_mapping_ignores_unknown_keys(self):
        options = on.VisionOptions.from_mapping({"max_tokens": 500, "unknown": 1})
        self.assertEqual(options.max_tokens, 500)

    def test_visual_result_never_contains_markdown(self):
        result = self.engine.convert(make_frame(dialog_frame(400, 260)))
        self.assertEqual(result.markdown, "")
        self.assertEqual(result.text_content, "")

    def test_detect_blocks_splits_layout(self):
        blocks = on.detect_blocks(text_page(800, 560))
        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(all(block.rect.area > 0 for block in blocks))
        self.assertGreaterEqual(blocks[0].salience, blocks[-1].salience)


# --------------------------------------------------------------------------- #
# 流式
# --------------------------------------------------------------------------- #


class StreamTest(unittest.TestCase):
    def test_identical_frames_are_deduped(self):
        engine = on.OpticNerve()
        stream = on.OpticNerveStream(engine)
        pixels = dialog_frame(400, 300)
        first = stream.push(make_frame(pixels, frame_id=1))
        second = stream.push(make_frame(pixels.copy(), frame_id=2, timestamp_ms=100))
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(stream.stats()["deduped"], 1)

    def test_changed_frame_is_emitted(self):
        engine = on.OpticNerve()
        stream = on.OpticNerveStream(engine)
        pixels = dialog_frame(400, 300)
        stream.push(make_frame(pixels, frame_id=1))
        changed = pixels.copy()
        changed[180:480, 260:760] = 30
        result = stream.push(make_frame(changed, frame_id=2, timestamp_ms=200))
        self.assertIsNotNone(result)
        self.assertEqual(stream.stats()["emitted"], 2)

    def test_stats_report_token_savings(self):
        engine = on.OpticNerve()
        stream = on.OpticNerveStream(engine)
        stream.push(make_frame(text_page(720, 520)))
        stats = stream.stats()
        self.assertGreater(stats["tokens_in"], stats["tokens_out"])
        self.assertLess(stats["token_ratio"], 1.0)
        self.assertEqual(stats["pushed"], 1)

    def test_engine_stream_generator(self):
        engine = on.OpticNerve()
        frames = [make_frame(dialog_frame(320, 240), frame_id=index, timestamp_ms=index * 50) for index in range(3)]
        results = list(engine.stream(frames))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item.kind is on.ResultKind.IMAGE for item in results))

    def test_observation_conversion(self):
        engine = on.OpticNerve()
        frame = make_frame(text_page(640, 440))
        observed = Observation(kind="frame", seq=1, source="desktop", frame=frame, dirty_rects=[Rect(46, 28, 200, 40)])
        self.assertIsNotNone(engine.convert_observation(observed))
        self.assertIsNone(engine.convert_observation(Observation(kind="unchanged", seq=2, source="desktop")))
        self.assertIsNone(engine.convert_observation(Observation(kind="silence", seq=3, source="desktop")))

    def test_observed_unchanged_via_convert(self):
        engine = on.OpticNerve()
        self.assertIsNone(engine.convert(Observation(kind="unchanged", seq=9, source="desktop")))


# --------------------------------------------------------------------------- #
# 结构化 / CSV / JSON
# --------------------------------------------------------------------------- #


class StructuredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = on.OpticNerve()

    def test_csv_preserves_columns_and_rows(self):
        data = b"name,score,note\nalice,91,\"hello, world\"\nbob,80,\"line1\"\n"
        result = self.engine.convert(data, stream_info=on.StreamInfo(extension=".csv", mimetype="text/csv"))
        self.assertIs(result.kind, on.ResultKind.STRUCTURED)
        self.assertEqual(result.structured.columns, ["name", "score", "note"])
        self.assertEqual(len(result.structured.rows), 2)
        self.assertEqual(result.structured.rows[0][2], "hello, world")
        self.assertIn("| name | score | note |", result.markdown)

    def test_tsv_delimiter_detected(self):
        data = b"a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        result = self.engine.convert(data, stream_info=on.StreamInfo(extension=".tsv", mimetype="text/tab-separated-values"))
        self.assertEqual(result.structured.columns, ["a", "b", "c"])
        self.assertEqual(result.structured.shape, (2, 3))

    def test_csv_numeric_cells_are_typed(self):
        result = self.engine.convert(b"n,v\n1,2.5\n2,3\n", stream_info=on.StreamInfo(extension=".csv"))
        self.assertEqual(result.structured.rows[0][0], 1)
        self.assertEqual(result.structured.rows[0][1], 2.5)

    def test_csv_truncation_flag(self):
        rows = ["id,val"] + [f"{index},{index * 2}" for index in range(10)]
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(on.StructuredConverter(max_rows=4), priority=10.0)
        result = engine.convert("\n".join(rows).encode("utf-8"), stream_info=on.StreamInfo(extension=".csv"))
        self.assertTrue(result.structured.truncated)
        self.assertLessEqual(len(result.structured.rows), 4)

    def test_json_records(self):
        payload = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).encode("utf-8")
        result = self.engine.convert(payload, stream_info=on.StreamInfo(extension=".json", mimetype="application/json"))
        self.assertIs(result.kind, on.ResultKind.STRUCTURED)
        self.assertEqual(len(result.structured.records), 2)
        self.assertIn("| a | b |", result.markdown)

    def test_json_object_becomes_key_value(self):
        payload = json.dumps({"name": "optic", "version": 2}).encode("utf-8")
        result = self.engine.convert(payload, stream_info=on.StreamInfo(extension=".json"))
        self.assertEqual(result.structured.format, "key_value")
        self.assertEqual(len(result.structured.rows), 2)

    def test_jsonl_stream(self):
        payload = b'{"a": 1}\n{"a": 2}\n'
        result = self.engine.convert(payload, stream_info=on.StreamInfo(extension=".jsonl"))
        self.assertEqual(len(result.structured.records), 2)

    def test_structured_tokens_use_compact_form(self):
        data = b"a,b\n1,2\n"
        result = self.engine.convert(data, stream_info=on.StreamInfo(extension=".csv"))
        self.assertGreater(result.tokens.original, 0)
        self.assertLessEqual(result.tokens.output, result.tokens.original)


# --------------------------------------------------------------------------- #
# 代码
# --------------------------------------------------------------------------- #


class CodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = on.OpticNerve()

    def test_python_code_is_fenced_with_outline(self):
        source = 'import os\n\n\ndef alpha(x):\n    return x\n\n\nclass Beta:\n    def run(self):\n        return 1\n'
        result = self.engine.convert(source.encode("utf-8"), stream_info=on.StreamInfo(extension=".py"))
        self.assertIs(result.kind, on.ResultKind.STRUCTURED)
        self.assertEqual(result.structured.format, "code")
        self.assertIn("```python", result.markdown)
        self.assertIn("def alpha", result.markdown)
        self.assertTrue(result.meta["outline"])

    def test_long_file_is_truncated(self):
        source = "\n".join(f"def fn_{index}():\n    return {index}" for index in range(400))
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(on.CodeConverter(max_lines=60), priority=12.0)
        result = engine.convert(source.encode("utf-8"), stream_info=on.StreamInfo(extension=".py"))
        self.assertLessEqual(len(result.markdown.splitlines()), 70)
        self.assertIn("省略", result.markdown)

    def test_code_tokens_never_exceed_source(self):
        source = "\n".join(f"value_{index} = {index}" for index in range(50))
        result = self.engine.convert(source.encode("utf-8"), stream_info=on.StreamInfo(extension=".py"))
        self.assertLessEqual(result.tokens.output, result.tokens.original + 5)


# --------------------------------------------------------------------------- #
# 视频
# --------------------------------------------------------------------------- #


class VideoTest(unittest.TestCase):
    def test_keyframes_are_sampled(self):
        frames = [photo_frame(160, 120, seed=index) for index in range(12)]
        converter = on.VideoConverter(max_frames=4, reader_factory=lambda _: FakeVideoReader(frames))
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(converter, priority=1.0)
        result = engine.convert(b"fake-video", stream_info=on.StreamInfo(extension=".mp4", mimetype="video/mp4"))
        self.assertIs(result.kind, on.ResultKind.FRAME)
        self.assertGreaterEqual(len(result.frames), 2)
        self.assertLessEqual(len(result.frames), 4)
        self.assertIn("fps", result.meta)

    def test_transcribe_hook_is_reported(self):
        frames = [photo_frame(160, 120, seed=index) for index in range(6)]
        converter = on.VideoConverter(
            max_frames=3,
            reader_factory=lambda _: FakeVideoReader(frames),
            transcribe=lambda _: "讲解：这是操作步骤",
        )
        engine = on.OpticNerve(enable_builtins=False)
        engine.register_converter(converter, priority=1.0)
        result = engine.convert(b"fake-video", stream_info=on.StreamInfo(extension=".mp4"))
        self.assertEqual(result.meta["transcript"], "讲解：这是操作步骤")
        self.assertEqual(result.title, "讲解：这是操作步骤")


# --------------------------------------------------------------------------- #
# 文档
# --------------------------------------------------------------------------- #


class DocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = on.OpticNerve()

    def test_plain_text(self):
        result = self.engine.convert(b"hello\n\nworld\n", stream_info=on.StreamInfo(extension=".txt", mimetype="text/plain"))
        self.assertIs(result.kind, on.ResultKind.MARKDOWN)
        self.assertEqual(result.markdown, "hello\n\nworld")

    def test_markdown_passthrough(self):
        data = "# 标题\n\n- 一\n- 二\n".encode("utf-8")
        result = self.engine.convert(data, stream_info=on.StreamInfo(extension=".md", mimetype="text/markdown"))
        self.assertIn("# 标题", result.markdown)
        self.assertEqual(result.title, "标题")

    def test_html_to_markdown(self):
        html = b"<html><head><title>T</title></head><body><h1>Big</h1><p>para <b>bold</b></p><ul><li>a</li><li>b</li></ul><table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>"
        result = self.engine.convert(html, stream_info=on.StreamInfo(extension=".html", mimetype="text/html"))
        self.assertIs(result.kind, on.ResultKind.MARKDOWN)
        self.assertIn("# Big", result.markdown)
        self.assertIn("**bold**", result.markdown)
        self.assertIn("- a", result.markdown)
        self.assertIn("| x | y |", result.markdown)

    def test_docx_heading_paragraph_and_table(self):
        result = self.engine.convert(docx_bytes(), stream_info=on.StreamInfo(extension=".docx"))
        self.assertIs(result.kind, on.ResultKind.MARKDOWN)
        self.assertIn("# Doc Title", result.markdown)
        self.assertIn("First paragraph", result.markdown)
        self.assertIn("| H1 | H2 |", result.markdown)
        self.assertIn("| a | b |", result.markdown)

    def test_xlsx_sheets(self):
        try:
            from openpyxl import Workbook
        except ImportError:  # pragma: no cover
            self.skipTest("openpyxl 未安装")
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "数据"
        sheet.append(["名称", "数量"])
        sheet.append(["螺栓", 12])
        buffer = io.BytesIO()
        workbook.save(buffer)
        result = self.engine.convert(buffer.getvalue(), stream_info=on.StreamInfo(extension=".xlsx"))
        self.assertIs(result.kind, on.ResultKind.STRUCTURED)
        self.assertIn("| 名称 | 数量 |", result.markdown)
        self.assertIn("螺栓", result.markdown)

    def test_zip_listing(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.txt", "hello")
            archive.writestr("b/c.txt", "world")
        result = self.engine.convert(buffer.getvalue(), stream_info=on.StreamInfo(extension=".zip"))
        self.assertIn("a.txt", result.markdown)
        self.assertIn("b/c.txt", result.markdown)

    def test_image_as_document_is_markdown_link(self):
        result = self.engine.convert(jpeg_bytes(dialog_frame(120, 80)), as_document=True)
        self.assertIs(result.kind, on.ResultKind.MARKDOWN)
        self.assertIn("![image](data:image/png;base64,", result.markdown)

    def test_image_default_path_stays_pixels(self):
        result = self.engine.convert(jpeg_bytes(dialog_frame(160, 100)))
        self.assertIs(result.kind, on.ResultKind.IMAGE)

    def test_utf8_charset_body(self):
        data = "中文内容，测试编码。".encode("utf-8")
        result = self.engine.convert(data, stream_info=on.StreamInfo(mimetype="text/plain", charset="utf-8"))
        self.assertIn("中文内容", result.markdown)


# --------------------------------------------------------------------------- #
# 与大脑的兼容
# --------------------------------------------------------------------------- #


class BrainCompatTest(unittest.TestCase):
    def test_image_bytes_feed_vision_brain(self):
        from vision_retina.brain import VisionBrain

        result = on.OpticNerve().convert(make_frame(text_page(600, 420)))
        brain = VisionBrain(base_url="http://127.0.0.1:9/v1", api_key="test")
        parts = [brain.image_part(item) for item in result.image_bytes()]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_openai_content_keeps_text_first(self):
        result = on.OpticNerve().convert(make_frame(dialog_frame(400, 260)), mode="dual")
        content = result.to_openai_content("看这里")
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(sum(1 for item in content if item["type"] == "image_url"), 2)

    def test_estimate_helpers_sane(self):
        self.assertGreater(on.estimate_image_tokens(1600, 1000), on.estimate_image_tokens(800, 500))
        self.assertGreater(on.estimate_text_tokens("abcdefgh"), 0)
        self.assertGreaterEqual(on.estimate_image_tokens(8, 8), 48)

    def test_default_engine_singleton(self):
        on.reset_default_engine()
        first = on.default_engine()
        self.assertIs(first, on.default_engine())
        on.reset_default_engine()
        self.assertIsNot(first, on.default_engine())

    def test_mapper_to_dict_is_serialisable(self):
        result = on.OpticNerve().convert(make_frame(dialog_frame(600, 400)))
        json.dumps(result.mapper.to_dict())
        json.dumps(result.to_dict(include_payload=False))


class ServiceIntegrationTests(unittest.TestCase):
    """视网膜服务层的接入：perceive() = 采一帧 → 视神经 → 喂大脑的形态。"""

    class _Bus:
        def __init__(self, observations):
            self._queue = list(observations)
            self.region = None

        def start(self):
            return self

        def close(self):
            pass

        def observe(self, force_keyframe=False):
            return self._queue.pop(0)

        def set_region(self, region):
            self.region = region

        def status(self):
            return {"mode": "delta"}

    def _service(self, observations):
        from vision_retina.config import load_config
        from vision_retina.service import RetinaService

        service = RetinaService(load_config())
        service._bus = self._Bus(observations)
        self.addCleanup(service.close)
        return service

    def _frame(self, frame_id=1):
        pixels = dialog_frame(600, 400)
        return RetinaFrame(
            frame_id=frame_id,
            timestamp_ms=1_700_000_000_000,
            source="desktop",
            pixels=pixels,
            dirty_rects=[Rect(40, 30, 260, 120)],
        )

    def test_perceive_returns_brain_ready_payload(self):
        service = self._service([
            Observation(kind="frame", seq=1, source="desktop", frame=self._frame(), raw_size=(600, 400)),
        ])
        payload = service.perceive()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["skipped"])
        self.assertEqual(payload["kind"], "image")
        self.assertEqual(payload["image"]["mime"], "image/jpeg")
        self.assertGreater(payload["image"]["bytes"], 0)
        self.assertNotIn("base64", payload["image"])
        self.assertLess(payload["tokens"]["output"], payload["tokens"]["original"])
        self.assertIsInstance(service.optic_nerve, on.OpticNerve)

    def test_perceive_can_embed_base64(self):
        service = self._service([
            Observation(kind="frame", seq=1, source="desktop", frame=self._frame(), raw_size=(600, 400)),
        ])
        payload = service.perceive(include_image=True)
        encoded = payload["image"]["base64"]
        self.assertTrue(encoded)
        self.assertTrue(on.decode_image(__import__("base64").b64decode(encoded)).size > 0)

    def test_perceive_skips_unchanged_frame(self):
        service = self._service([
            Observation(kind="unchanged", seq=2, source="desktop", frame=None, raw_size=(600, 400),
                        reason="无变化"),
        ])
        payload = service.perceive()
        self.assertTrue(payload["skipped"])
        self.assertEqual(payload["reason"], "unchanged")
        self.assertNotIn("image", payload)
        self.assertTrue(payload["frame_id"] is None or payload["frame_id"] >= 0)

    def test_optic_nerve_property_is_cached(self):
        service = self._service([])
        self.assertIs(service.optic_nerve, service.optic_nerve)

    def test_mcp_server_exposes_perceive_tool(self):
        service = self._service([])
        mcp_server = __import__("vision_retina.mcp_server", fromlist=["build_mcp_server"])
        server = mcp_server.build_mcp_server(service)
        names = [tool.name for tool in server._tool_manager.list_tools()]
        self.assertIn("retina_perceive", names)
        self.assertIn("retina_observe", names)


if __name__ == "__main__":
    unittest.main()
