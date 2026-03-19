import os
import json
import base64
import io
import tempfile
import time
import queue
import threading
import gradio as gr
import gradio.blocks
from PIL import Image
from rag.graph import analyze_online
from dotenv import load_dotenv
load_dotenv()

# 修復 gradio bug
gradio.blocks.Blocks.get_api_info = lambda self: {"named_endpoints": {}, "unnamed_endpoints": {}}


# ==============================
# Agent 狀態列 → Markdown
# ==============================

AGENT_LABELS = {
    "ocr":       "OCR 辨識",
    "normalize": "名稱標準化",
    "parser":    "成分解析",
    "query":     "資料庫查詢",
    "enrich":    "AI 增強",
    "response":  "整合輸出",
}

STATUS_ICONS = {
    "pending": "⬜",
    "running": "⏳",
    "done":    "✅",
    "skip":    "—",
}

def format_agent_status(agent_status: dict) -> str:
    parts = []
    for node, status in agent_status.items():
        if status == "skip":
            continue
        icon = STATUS_ICONS.get(status, "?")
        label = AGENT_LABELS.get(node, node)
        parts.append(f"{icon} {label}")

    if not parts:
        return ""

    pipeline_str = " → ".join(parts)
    return f"**Agent Pipeline**\n\n{pipeline_str}"


# ==============================
# 成分卡片 → Markdown 格式
# ==============================

def format_card_md(item: dict) -> str:
    name = item.get("_display_name") or item.get("_original") or item.get("ingredient") or item.get("inci_name", "Unknown")
    confidence = item.get("confidence", "medium").lower()

    if confidence == "error":
        return f"### ❌ {item.get('ingredient', 'unknown')}\n查詢失敗\n\n---"

    badge = "🟢 官方資料" if confidence == "high" else "🟡 AI 推論"

    inci      = item.get("inci_name", "")
    cas       = item.get("cas_number", "")
    functions = item.get("functions", [])
    benefits  = item.get("benefits", [])
    risks     = item.get("risks", [])
    eu_reg    = item.get("eu_regulation", "")
    skin      = item.get("skin_type", [])
    warning   = item.get("warning", "")

    lines = [f"### {name} &nbsp; {badge}"]

    meta_parts = []
    if inci:
        meta_parts.append(f"INCI: `{inci}`")
    if cas:
        meta_parts.append(f"CAS: `{cas}`")
    if meta_parts:
        lines.append(" &nbsp;|&nbsp; ".join(meta_parts))

    if functions:
        lines.append(f"**功能：** {', '.join(functions)}")
    if benefits:
        lines.append("\n**功效**")
        for b in benefits:
            lines.append(f"- {b}")
    if risks:
        lines.append("\n**風險**")
        for r in risks:
            lines.append(f"- ⚠️ {r}")
    if skin:
        lines.append(f"\n**適合膚質：** {', '.join(skin)}")
    if eu_reg:
        lines.append(f"\n**EU 法規：** {eu_reg}")
    if warning:
        lines.append(f"\n> ⚠️ {warning}")

    lines.append("\n---")
    return "\n".join(lines)


def process_results_md(results: list) -> str:
    if not results:
        return "### ⚠️ 未找到任何成分"
    return "\n\n".join(format_card_md(r) for r in results)


# ==============================
# 比較結果
# ==============================

def build_compare_outputs(results1, results2, source_label="文字輸入"):
    def extract_names(results):
        return {
            (r.get("_display_name") or r.get("_original") or r.get("ingredient", "")).strip().lower()
            for r in results if not r.get("error")
        }

    def risk_score(results):
        scores = [len(r.get("risks", [])) for r in results if r.get("risks")]
        return sum(scores) / len(scores) if scores else 0.0

    names1 = extract_names(results1)
    names2 = extract_names(results2)

    common = sorted(names1 & names2)
    only1  = sorted(names1 - names2)
    only2  = sorted(names2 - names1)
    risk1  = risk_score(results1)
    risk2  = risk_score(results2)

    summary_md = (
        f"## 📊 比較結果總覽（{source_label}）\n\n"
        f"| | 產品 A | 產品 B |\n"
        f"|---|---|---|\n"
        f"| 成分數量 | {len(names1)} | {len(names2)} |\n"
        f"| 平均風險分數 | {risk1:.1f} | {risk2:.1f} |\n"
        f"| 相同成分 | {len(common)} 種 | — |\n"
        f"| 獨有成分 | {len(only1)} 種 | {len(only2)} 種 |"
    )

    df_common = [[i.title()] for i in common] if common else [["（無相同成分）"]]
    df_only1  = [[i.title()] for i in only1]  if only1  else [["（無獨有成分）"]]
    df_only2  = [[i.title()] for i in only2]  if only2  else [["（無獨有成分）"]]

    if risk1 < risk2:
        advice = "✅ **產品 A** 整體風險較低，成分較溫和。"
    elif risk2 < risk1:
        advice = "✅ **產品 B** 整體風險較低，成分較溫和。"
    else:
        advice = "兩個產品風險相當。"

    if len(names1) != len(names2):
        advice += f"  \n成分數量差異明顯（A: {len(names1)} 種 vs B: {len(names2)} 種）。"

    if source_label != "文字輸入":
        advice += "  \n_注意：此比較基於 OCR 辨識結果，準確度取決於圖片品質。_"

    advice_md = f"## 💡 建議\n\n{advice}"
    return summary_md, df_common, df_only1, df_only2, advice_md


# ==============================
# 文字分析（threading + queue 即時更新）
# ==============================

def analyze_text(ingredient_input: str):
    if not ingredient_input.strip():
        yield "### 💡 請輸入成分名稱", "", gr.DownloadButton(visible=False)
        return

    status_queue = queue.Queue()
    result_container = {}

    def run():
        def on_status(status):
            status_queue.put(("status", status))
            result_container["last_status"] = status

        try:
            print(f"[APP] 開始分析文字輸入：{ingredient_input[:100]}")
            results = analyze_online(ingredient_input.strip(), status_callback=on_status)
            result_container["results"] = results
        except Exception as e:
            print(f"[APP] 分析失敗：{e}")
            result_container["error"] = str(e)
        finally:
            status_queue.put(("done", None))

    thread = threading.Thread(target=run)
    thread.start()

    # 即時 yield 每次狀態更新
    while True:
        msg_type, payload = status_queue.get()
        if msg_type == "done":
            break
        if msg_type == "status":
            yield "⏳ 分析中...", format_agent_status(payload), gr.DownloadButton(visible=False)

    thread.join()

    if "error" in result_container:
        yield f"### ❌ 發生錯誤\n\n`{result_container['error']}`", "", gr.DownloadButton(visible=False)
        return

    results = result_container["results"]
    last_status = result_container.get("last_status", {})

    pending = []
    for r in results:
        if r.get("confidence") == "medium" and r.get("source") == ["LLM-generated"]:
            clean = {k: v for k, v in r.items() if k not in ("_query", "_original", "is_substance")}
            pending.append(clean)

    if pending:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(pending, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        download_btn = gr.DownloadButton(visible=True, value=tmp.name)
    else:
        download_btn = gr.DownloadButton(visible=False)

    final_status = {k: ("done" if v != "skip" else "skip") for k, v in last_status.items()}

    print(f"[APP] 文字分析完成，返回 {len(results)} 筆結果")
    yield process_results_md(results), format_agent_status(final_status), download_btn


# ==============================
# 圖片分析（threading + queue 即時更新）
# ==============================

def analyze_image(files):
    if not files:
        yield "### 💡 請上傳圖片", ""
        return

    status_queue = queue.Queue()
    result_container = {}

    def run():
        def on_status(status):
            status_queue.put(("status", status))
            result_container["last_status"] = status

        try:
            file = files[0]
            print(f"[APP] 處理上傳的圖片：{file.name}")

            image = Image.open(file.name)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            b64 = base64.b64encode(buffered.getvalue()).decode()

            results = analyze_online("", image_b64=b64, status_callback=on_status)
            result_container["results"] = results
        except FileNotFoundError:
            result_container["error"] = "圖片文件未找到"
        except Exception as e:
            print(f"[APP] 圖片處理失敗：{e}")
            result_container["error"] = str(e)
        finally:
            status_queue.put(("done", None))

    thread = threading.Thread(target=run)
    thread.start()

    while True:
        msg_type, payload = status_queue.get()
        if msg_type == "done":
            break
        if msg_type == "status":
            yield "⏳ 分析中...", format_agent_status(payload)

    thread.join()

    if "error" in result_container:
        yield f"### ❌ 圖片處理失敗\n\n`{result_container['error']}`", ""
        return

    results = result_container["results"]
    last_status = result_container.get("last_status", {})
    final_status = {k: ("done" if v != "skip" else "skip") for k, v in last_status.items()}

    yield process_results_md(results), format_agent_status(final_status)


# ==============================
# 文字比較
# ==============================

def compare_products(product1: str, product2: str, progress=gr.Progress()):
    empty = ("### 💡 請輸入兩個產品的成分進行比較", [], [], [], "")
    if not product1.strip() or not product2.strip():
        return empty

    try:
        progress(0.0, desc="分析產品A...")
        results1 = analyze_online(product1.strip())
        progress(0.5, desc="分析產品B...")
        results2 = analyze_online(product2.strip())
        progress(1.0, desc="比較分析...")
        summary, df_common, df_only1, df_only2, advice = build_compare_outputs(results1, results2)
        return summary, df_common, df_only1, df_only2, advice
    except Exception as e:
        print(f"[COMPARE] 比較失敗：{e}")
        return f"### ❌ 比較失敗\n\n`{str(e)}`", [], [], [], ""


# ==============================
# 圖片比較
# ==============================

def compare_products_from_images(files1, files2, progress=gr.Progress()):
    empty = ("### 💡 請上傳兩個產品的成分表圖片", [], [], [], "")
    if not files1 or not files2:
        return empty

    try:
        def load_b64(files):
            img = Image.open(files[0].name)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode()

        progress(0.0, desc="處理產品A圖片...")
        b64_1 = load_b64(files1)
        progress(0.2, desc="OCR辨識產品A...")
        results1 = analyze_online("", image_b64=b64_1)

        progress(0.4, desc="處理產品B圖片...")
        b64_2 = load_b64(files2)
        progress(0.6, desc="OCR辨識產品B...")
        results2 = analyze_online("", image_b64=b64_2)

        progress(0.9, desc="比較分析...")
        summary, df_common, df_only1, df_only2, advice = build_compare_outputs(
            results1, results2, source_label="圖片辨識"
        )
        progress(1.0, desc="完成")
        return summary, df_common, df_only1, df_only2, advice

    except Exception as e:
        print(f"[COMPARE_IMG] 圖片比較失敗：{e}")
        return f"### ❌ 圖片比較失敗\n\n`{str(e)}`", [], [], [], ""


# ==============================
# UI
# ==============================

with gr.Blocks(
    title="Cosmetic Ingredient Analyzer",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown("CosIng + RAG + LangGraph Multi-Agent 成分分析系統")

    with gr.Tabs():

        # ── Tab 1：文字分析 ──────────────────────────────────
        with gr.Tab("文字分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="輸入成分",
                        placeholder="Niacinamide, Retinol, Glycerin",
                        lines=4
                    )
                    text_btn = gr.Button("開始分析", variant="primary")
                    download_btn = gr.DownloadButton(
                        label="下載 AI 成分資料",
                        visible=False
                    )

                with gr.Column(scale=2):
                    text_agent_status = gr.Markdown(value="")      # ← 右欄頂部，初始為空
                    text_output = gr.Markdown(value="等待輸入...")

            gr.Examples(
                examples=[
                    ["Salicylic Acid, Glycerin"],
                    ["Niacinamide, Hyaluronic Acid"],
                    ["Retinol"],
                    ["Bakuchiol"],
                ],
                inputs=text_input
            )

        # ── Tab 2：圖片辨識 ──────────────────────────────────
        with gr.Tab("圖片辨識"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.File(
                        label="上傳成分表",
                        file_types=["image"],
                        file_count="multiple"
                    )
                    image_btn = gr.Button("辨識並分析", variant="primary")

                with gr.Column(scale=2):
                    image_agent_status = gr.Markdown(value="")     # ← 右欄頂部，初始為空
                    image_output = gr.Markdown(value="請上傳圖片")

        # ── Tab 3：成分比較 ──────────────────────────────────
        with gr.Tab("成分比較"):
            with gr.Tabs():

                # 文字比較
                with gr.TabItem("文字比較"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            product1_text = gr.Textbox(
                                label="產品A成分",
                                placeholder="Niacinamide, Hyaluronic Acid, Glycerin",
                                lines=3
                            )
                            product2_text = gr.Textbox(
                                label="產品B成分",
                                placeholder="Retinol, Vitamin C, Salicylic Acid",
                                lines=3
                            )
                            compare_text_btn = gr.Button("比較產品", variant="primary")

                    compare_text_summary = gr.Markdown(value="輸入兩個產品的成分進行比較")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🔄 相同成分")
                            compare_text_common = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )
                        with gr.Column():
                            gr.Markdown("#### ➡️ 僅在產品 A")
                            compare_text_only1 = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )
                        with gr.Column():
                            gr.Markdown("#### ⬅️ 僅在產品 B")
                            compare_text_only2 = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )

                    compare_text_advice = gr.Markdown()

                    gr.Examples(
                        examples=[
                            ["Niacinamide, Hyaluronic Acid", "Retinol, Vitamin C"],
                            ["Glycerin, Panthenol", "Propylene Glycol, PEG-400"],
                            ["Salicylic Acid, Benzoyl Peroxide", "Tea Tree Oil, Witch Hazel"],
                        ],
                        inputs=[product1_text, product2_text]
                    )

                # 圖片比較
                with gr.TabItem("圖片比較"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            product1_image = gr.File(
                                label="產品A成分表",
                                file_types=["image"],
                                file_count="multiple"
                            )
                            product2_image = gr.File(
                                label="產品B成分表",
                                file_types=["image"],
                                file_count="multiple"
                            )
                            compare_image_btn = gr.Button("比較產品", variant="primary")

                    compare_image_summary = gr.Markdown(value="上傳兩個產品的成分表圖片進行比較")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🔄 相同成分")
                            compare_image_common = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )
                        with gr.Column():
                            gr.Markdown("#### ➡️ 僅在產品 A")
                            compare_image_only1 = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )
                        with gr.Column():
                            gr.Markdown("#### ⬅️ 僅在產品 B")
                            compare_image_only2 = gr.Dataframe(
                                headers=["成分"], datatype=["str"],
                                interactive=False, wrap=True
                            )

                    compare_image_advice = gr.Markdown()

    gr.Markdown("---\n資料來源：CosIng / INCI Decoder  \nAI 生成資料僅供參考")

    # ── 事件綁定 ────────────────────────────────────────────
    text_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[text_output, text_agent_status, download_btn]
    )
    text_input.submit(
        fn=analyze_text,
        inputs=text_input,
        outputs=[text_output, text_agent_status, download_btn]
    )
    image_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[image_output, image_agent_status]
    )
    compare_text_btn.click(
        fn=compare_products,
        inputs=[product1_text, product2_text],
        outputs=[
            compare_text_summary, compare_text_common,
            compare_text_only1, compare_text_only2, compare_text_advice,
        ]
    )
    compare_image_btn.click(
        fn=compare_products_from_images,
        inputs=[product1_image, product2_image],
        outputs=[
            compare_image_summary, compare_image_common,
            compare_image_only1, compare_image_only2, compare_image_advice,
        ]
    )


if __name__ == "__main__":
    if not os.path.exists("faiss_index/index.faiss"):
        raise FileNotFoundError("找不到 FAISS 索引")

    is_hf = os.environ.get("SPACE_ID") is not None

    demo.queue()          # ← 串流更新必要
    demo.launch(
        server_name="0.0.0.0" if is_hf else "127.0.0.1"
        )