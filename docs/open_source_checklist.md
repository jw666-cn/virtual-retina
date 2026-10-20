# 开源发布检查清单（脱敏红线）

> 检查日期：2026-10-20。检查对象：本地工作区 `<repo-root>`。
> 结论：**代码与文档可开源，但必须先把密钥、artifacts、个人路径/示例名处理掉**；当前目录不能直接打包发布。

## 1. 红线：绝不公开

| 路径 / 内容 | 现状 | 处置 |
|---|---|---|
| `config/secrets.local.env` | **存在，121 B，含 `VISION_API_KEY` 配置（已 `.gitignore` 排除，不入库）** | 发布包不包含；该 Key 曾在 OneDrive 工作区存在，**建议到控制台轮换/作废** |
| 任何 `*.local.env`、`.env` | `.gitignore` 已忽略 `config/secrets.local.env` 与 `*.local.env` | 打包前确认不存在；不要用 `zip -r .` 直接打包 |
| `artifacts/` 全部（329 文件，24.5 MB） | 含 DSH 界面截图（账户余额、费用、头像）、CAD 原图与复绘、动态测试画面、模型对比帧、OCR 采样 | **整目录剔除**；`.gitignore` 已忽略 |
| `artifacts/cad_src/*.jpg` | 外部 CAD 图纸（有版权来源） | 不公开；复绘派生图也不随仓库发布 |
| `artifacts/ocr_compare/real_ui.png`、`artifacts/dynamic/*.png` | 真实桌面截图，含账号/费用信息 | 不公开 |
| `tools/_glm_probe.py`、`tools/_glm_probe2.py` | 读取环境变量 Key，无硬编码，但属一次性厂商探针 | 建议移出公开包或放 `tools/dev/` |
| 任何账单金额、发票、账号 ID | 仓库正文未发现金额；金额只出现在截图中 | 截图不公开；报告只保留 token/延迟 |
| `__pycache__/`、`_tmp/`、`*.log` | `.gitignore` 已覆盖；`_tmp/` 已删除 | 打包前清理 |

## 2. 可公开

| 路径 | 说明 |
|---|---|
| `vision_retina/**` | 产品代码（capture / bus / bionic / adaptive / roi / brain / service / MCP / WS / CLI） |
| `tests/**` | 27 项单测，无密钥、无外部素材 |
| `tools/**`（除 `_glm_probe*.py`） | 自检/基准/闭环脚本；无硬编码 Key |
| `config/retina.yaml` | 仅非敏感参数；`brain.base_url` 为空、Key 走 secrets/环境变量 |
| `config/secrets.local.env.example` | 占位模板，可公开 |
| `README.md` | 需先替换本机绝对路径（见 §3） |
| `docs/接口协议.md`、`docs/biomimicry.md`、`docs/CAPABILITIES.md`、`docs/Phase3-跨平台与IDD方案.md` | 架构与协议，可公开 |
| `reports/fovea_budget.md`、`reports/adaptive.md`、`reports/cu_compare.md`、`reports/model_compare.md`、`reports/ocr_compare.md` | 只有指标/结论；`model_compare` 的“费用”列是定性描述，无金额 |
| `docs/验收报告.md`、`docs/动态实时性测试报告.md`、`docs/动态测试修复报告.md`、`docs/复绘对照报告.md` | 可公开，但需替换示例用户名（见 §3） |
| `requirements.txt`、`.gitignore` | 可公开 |

## 3. 需脱敏 / 改写（低风险但不该带出去）

| 文件:行 | 内容 | 建议 |
|---|---|---|
| `README.md:47` | `"cwd": "C:\\Users\\<user>\\...\\<repo>"` | 改为 `"<path-to-repo>"` |
| `docs/接口协议.md:53` | 示例动作 `"text": "<sample-user>"` | 改为 `"demo"` |
| `docs/验收报告.md:52` | 受控任务输入 `<sample-user>` | 改为 `demo` |
| `tools/gui_target.py:31` | 提示“填入 <sample-user>” | 改为 `demo` |
| `tools/closed_loop.py:65` | `--expect` 默认 `<sample-user>` | 改为 `demo` |
| `tools/acceptance_report.py:164` | 报告模板里复述 `<sample-user>` | 改为 `demo` |
| `docs/验收报告.md` 全文 | 本机路径 / 运行时间 / 模型名 | 模型名与指标可留；本机路径建议删除 |

## 4. 当前扫描结果

- `git`：**尚未初始化**（`fatal: not a git repository`）；`.gitignore` 已存在且覆盖 secrets / artifacts / 临时文件。
- 密钥扫描：**命中 1 处真实 Key**（`config/secrets.local.env`），其余 `.py` 仅引用 `api_key` 变量，无硬编码。
- 金额扫描：仓库正文未发现 `¥` / 账单金额；`reports/model_compare.md` 的“费用”仅为“按官方账单计费 / 免费档”定性描述。
- 个人路径：`README.md` 1 处绝对路径。
- 示例用户名：6 处（见 §3）。
- 已清理：`artifacts/ocr_compare/_edgeprofile`、`_ep2`（Edge 缓存，含浏览数据）；`_tmp/` 删除后可能被残留进程重建，发布前再清一次。

## 5. 发布前操作顺序

```powershell
# 1) 确认密钥文件不在发布树
Test-Path config/secrets.local.env        # 期望 False（或打包时排除）

# 2) 初始化并只 add 白名单
git init
git add vision_retina tests tools config/retina.yaml config/secrets.local.env.example `
        README.md docs reports requirements.txt .gitignore
git status --short

# 3) 两道扫描：密钥与金额
git ls-files | Select-String 'secrets|\.env$|artifacts/'
git grep -nE 'sk-[A-Za-z0-9]{20,}|API_KEY\s*=\s*[A-Za-z0-9]|¥|余额|账单'

# 4) 单测
py -3.12 -m unittest discover -s tests

# 5) 打包（显式排除，不要 zip -r .）
```

## 6. 许可与合规

- 仓库当前**没有 LICENSE**；公开发布前需选定许可证（MIT / Apache-2.0 等），本清单不替客户做法律选择。
- CAD 原图与复绘派生图均不随仓库发布；报告中只保留尺寸/识别结论。
- 第三方模型名（DeepSeek / GLM / UI-TARS）仅作事实引用，不包含其代码或权重。

## 7. 发布判定

| 项 | 状态 |
|---|---|
| 代码可运行、27 项单测通过 | ✅ |
| 无硬编码密钥 | ✅（唯一真实 Key 在 gitignore 的本地文件，需轮换） |
| 无金额/账单 | ✅（正文）；截图需剔除 |
| 无版权测试原图 | ✅（artifacts 整目录剔除） |
| 本机路径/示例名已替换 | ✅ 已执行（README / 接口协议 / 验收报告 / 3 个 tools） |
| LICENSE | ⏳ 待定 |

**结论：代码与正文已可开源；发布前只需确认 `config/secrets.local.env` 不进入发布包、剔除 `artifacts/`，并轮换该 Key。**
