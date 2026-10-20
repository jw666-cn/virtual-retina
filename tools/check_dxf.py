import json
from pathlib import Path
import ezdxf
from ezdxf import bbox

out = {}
for name in ["减速器装配图_复绘.dxf", "教学楼一层平面图_复绘.dxf"]:
    p = Path("artifacts/cad_redraw") / name
    doc = ezdxf.readfile(p)
    auditor = doc.audit()
    msp = doc.modelspace()
    ext = bbox.extents(msp)
    types = {}
    for e in msp:
        types[e.dxftype()] = types.get(e.dxftype(), 0) + 1
    out[name] = {
        "bytes": p.stat().st_size,
        "version": doc.dxfversion,
        "entities": len(msp),
        "types": types,
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        "extents_mm": [round(ext.extmin.x, 1), round(ext.extmin.y, 1), round(ext.extmax.x, 1), round(ext.extmax.y, 1)],
        "layers": [l.dxf.name for l in doc.layers],
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
Path("artifacts/cad_redraw/dxf_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
