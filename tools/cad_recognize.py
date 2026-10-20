import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json, time, sys
from pathlib import Path
from vision_retina.config import load_config
from vision_retina.brain import VisionBrain, BrainError

PROMPT = """你是工程图纸识别助手。请仔细识别这张 CAD 图纸，并只输出一个 JSON 对象（不要 markdown 代码块、不要解释），字段如下：
{
  "drawing_type": "图纸类型",
  "views": ["视图1", "视图2"],
  "dimensions": ["尺寸标注原文，如 612、Ø45H7/r6"],
  "tolerances": ["公差/配合"],
  "parts": [{"no": "序号或编号", "name": "名称"}],
  "texts": {"title": "标题", "scale": "比例", "notes": ["技术要求/表格文字"]},
  "uncertainties": ["看不清或不确定的内容"]
}
看不清的写“不可辨”，不要编造。"""

out = Path("artifacts/cad_rec"); out.mkdir(parents=True, exist_ok=True)
cfg = load_config()
brain = VisionBrain(cfg)
print("brain:", brain.status().get("endpoint"), brain.status().get("model"))
for key, path in [("gearbox", "artifacts/cad_src/gearbox.jpg"), ("building", "artifacts/cad_src/building.jpg")]:
    data = Path(path).read_bytes()
    t0 = time.perf_counter()
    try:
        reply = brain.ask(data, PROMPT, max_tokens=8192, temperature=0.0)
    except BrainError as e:
        print(key, "ERROR", type(e).__name__, e); continue
    ms = int((time.perf_counter()-t0)*1000)
    rec = {"image": path, "latency_ms": ms, "usage": reply.usage, "finish_reason": reply.finish_reason,
           "text": reply.text, "action": reply.action}
    (out/f"retina_{key}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{key}: {ms} ms usage={reply.usage} finish={reply.finish_reason}")
    print(reply.text[:1200].replace("\n", " "))
    print("-"*60)


