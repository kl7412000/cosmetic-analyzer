import os
import base64
import io
import gradio as gr
from PIL import Image
from rag.graph import analyze_online

custom_css = """
body, .gradio-container { background-color: #1a1a2e !important; color: #e0e0e0 !important; }
.ingredient-card {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.ingredient-card h3 { color: #e2e8f0 !important; margin-bottom: 8px; }
.ingredient-card p, .ingredient-card li { color: #c8c8c8 !important; }
.tag-high {
    color: #69f0ae !important;
    font-weight: 700;
    background: #1b3a2a;
    padding: 2px 10px;
    border-radius: 4px;
}
.tag-medium {
    color: #ffb74d !important;
    font-weight: 700;
    background: #3a2a10;
    padding: 2px 10px;
    border-radius: 4px;
}
"""


def format_card(item: dict) -> str:
    name = item.get("_original") or item.get("ingredient") or item.get("inci_name", "Unknown")
    confidence = item.get("confidence", "medium").lower()
    tag_class = "tag-high" if confidence == "high" else "tag-medium"
    tag_text = "✅ 官方數據庫" if confidence == "high" else "🤖 AI 生成"

    inci = item.get("inci_name", "")
    cas = item.get("cas_number", "")
    functions = item.get("functions", [])
    benefits = item.get("benefits", [])
    risks = item.get("risks", [])
    eu_reg = item.get("eu_regulation", "")
    skin = item.get("skin_type", [])
    warning = item.get("warning", "")

    md = f"<div class='ingredient-card'>\n\n"
    md += f"### 🔍 {name} &nbsp; <span class='{tag_class}'>{tag_text}</span>\n\n"
    if inci:
        md += f"**INCI 名稱：** {inci}　"
    if cas:
        md += f"**CAS：** {cas}\n\n"
    else:
        md += "\n\n"
    if functions:
        md += f"**功能分類：** {', '.join(functions)}\n\n"
    if benefits:
        md += "**功效：**\n"
        for b in benefits:
            md += f"- {b}\n"
        md += "\n"
    if risks:
        md += "**風險注意：**\n"
        for r in risks:
            md += f"- ⚠️ {r}\n"
        md += "\n"
    if skin:
        md += f"**適合膚質：** {', '.join(skin)}\n\n"
    if eu_reg:
        md += f"**歐盟法規：** {eu_reg}\n\n"
    if warning:
        md += f"> ⚠️ {warning}\n\n"
    md += "</div>\n\n"
    return md


def process_results(results: list) -> str:
    if not results:
        return "### ⚠️ 未找到任何成分資訊"
    output = ""
    for r in results:
        if r.get("error") and r.get("confidence") == "error":
            output += f"<div class='ingredient-card'>\n\n### ❌ {r.get('ingredient', 'unknown')}\n\n查詢失敗\n\n</div>\n\n"
        else:
            output += format_card(r)
    return output


def analyze_text(ingredient_input: str) -> str:
    if not ingredient_input.strip():
        return "### 💡 請輸入成分名稱"
    try:
        results = analyze_online(ingredient_input.strip())
        return process_results(results)
    except Exception as e:
        return f"### ❌ 發生錯誤\n`{str(e)}`"


def analyze_image(files) -> str:
    if not files:
        return "### 💡 請上傳圖片檔案"
    try:
        file = files[0]
        image = Image.open(file.name)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        results = analyze_online("", image_b64=b64)
        return process_results(results)
    except Exception as e:
        return f"### ❌ 圖片處理失敗\n`{str(e)}`"


with gr.Blocks(title="Cosmetic Ingredient Analyzer", css=custom_css,
               theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown("整合 **CosIng 數據庫** 與 **RAG 技術**，提供專業的成分分析報告。")

    # ── 文字分析 ──────────────────────────────────────────────────────────────
    gr.Markdown("## 📝 文字分析")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="請輸入成分（英文）",
                placeholder="多個成分請用逗號分隔\n例如：Niacinamide, Retinol, Glycerin",
                lines=4
            )
            text_btn = gr.Button("開始分析", variant="primary")
        with gr.Column(scale=2):
            text_output = gr.Markdown(value="等待輸入...")

    gr.Examples(
        examples=[
            ["Salicylic Acid, Glycerin"],
            ["Niacinamide, Hyaluronic Acid, Ceramide NP"],
            ["Retinol, Fragrance"],
            ["Bakuchiol"],
        ],
        inputs=text_input,
    )

    gr.Markdown("---")

    # ── 圖片辨識 ──────────────────────────────────────────────────────────────
    gr.Markdown("## 📷 圖片辨識")
    gr.Markdown("> 上傳化妝品成分標籤圖片，系統會自動辨識成分列表並分析（支援 JPG、PNG、WEBP）")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(
                label="上傳成分表照片",
                file_types=["image"],
                file_count="multiple",
            )
            image_btn = gr.Button("辨識並分析", variant="primary")
        with gr.Column(scale=2):
            image_output = gr.Markdown(value="請上傳圖片以開始分析...")

    gr.Markdown(
        "---\n"
        "**資料來源**：CosIng（歐盟官方）、INCI Decoder\n\n"
        "💡 **提示**：標記為 `AI 生成` 的資料由 LLM 即時生成，建議參考專業來源自行查證。"
    )

    text_btn.click(fn=analyze_text, inputs=text_input, outputs=text_output)
    text_input.submit(fn=analyze_text, inputs=text_input, outputs=text_output)
    image_btn.click(fn=analyze_image, inputs=image_input, outputs=image_output)

import gradio.blocks
gradio.blocks.Blocks.get_api_info = lambda self: {"named_endpoints": {}, "unnamed_endpoints": {}}

if __name__ == "__main__":
    if not os.path.exists("faiss_index/index.faiss"):
        raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")
    is_hf = os.environ.get("SPACE_ID") is not None
    demo.launch(server_name="0.0.0.0" if is_hf else "127.0.0.1")