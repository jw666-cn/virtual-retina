import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_retina.frames import Rect
from vision_retina.roi import expand_rect_to_content, group_lines


def canvas_with_label():
    img = np.zeros((120, 320, 3), dtype=np.uint8)
    # one text line made of 5 glyph boxes, x=50..250, y=40..70
    for i in range(5):
        x = 50 + i * 40
        img[40:70, x:x + 28] = 230
    return img


class RoiTests(unittest.TestCase):
    def test_expand_includes_whole_text_line(self):
        img = canvas_with_label()
        cut = Rect(115, 35, 25, 40)  # cuts glyph #2 and #3
        out, ok = expand_rect_to_content(img, cut)
        self.assertTrue(ok)
        self.assertLessEqual(out.x, 44)
        self.assertGreaterEqual(out.x + out.w, 244)
        self.assertLessEqual(out.y, 40)
        self.assertGreaterEqual(out.y + out.h, 70)

    def test_area_cap_falls_back(self):
        img = canvas_with_label()
        out, ok = expand_rect_to_content(img, Rect(115, 35, 25, 40), max_area_ratio=0.001)
        self.assertFalse(ok)
        self.assertEqual((out.x, out.y, out.w, out.h), (0, 0, img.shape[1], img.shape[0]))

    def test_group_lines_keeps_text_together(self):
        boxes = [(50 + i * 40, 40, 28, 30) for i in range(5)]
        lines = group_lines(boxes)
        self.assertEqual(len(lines), 1)
        x, y, w, h = lines[0]
        self.assertEqual((x, y, w, h), (50, 40, 188, 30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
