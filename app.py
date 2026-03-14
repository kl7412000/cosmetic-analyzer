import os
import json
import base64
import gradio as gr
from rag.graph import analyze_online

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if not os.path.exists("faiss_index/index.faiss"):
    raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")


def format_result(item: dict) -> str:
    display = {k: v for k, v in item.items()
               if k not in ("_query", "is_substance")}
    return json.dumps(display, ensure_ascii=False, indent=2)


def analyze_text(ingredient_input: str) -> str:
    if not ingredient_input.strip():
        return "請輸入成分名稱"
    try:
        results = analyze_online(ingredient_input.strip())
        if not results:
            return "無結果"
        if len(results) == 1:
            return format_result(results[0])
        output = []
        for r in results:
            name = r.get("ingredient") or r.get("inci_name", "unknown")
            output.append(f"// ── {name} ──")
            output.append(format_result(r))
        return "\n\n".join(output)
    except Exception as e:
        return f"發生錯誤：{e}"


def analyze_image(file) -> str:
    if file is None:
        return "請上傳圖片"

    try:
        path = file.name

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        name_lower = path.lower()

        if name_lower.endswith(".png"):
            media_type = "image/png"
        elif name_lower.endswith(".webp"):
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

        results = analyze_online("", image_b64=b64)

        if not results:
            return "圖片中未偵測到成分列表"

        if len(results) == 1:
            return format_result(results[0])

        output = []
        for r in results:
            name = r.get("ingredient") or r.get("inci_name", "unknown")
            output.append(f"// ── {name} ──")
            output.append(format_result(r))

        return "\n\n".join(output)

    except Exception as e:
        return f"圖片處理失敗：{e}"


with gr.Blocks(title="Cosmetic Ingredient Analyzer",
    analytics_enabled=False
) as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown(
        "輸入化妝品成分名稱或上傳成分標籤圖片，獲取功效、風險、歐盟法規資訊。\n\n"
        "- **已知成分**：從知識庫查詢（`confidence: high`）\n"
        "- **未知成分**：由 AI 即時生成（`confidence: medium`，附警告）"
    )

    with gr.Tabs():

        with gr.Tab("📝 文字輸入"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="成分名稱",
                        placeholder="輸入一個或多個成分，用逗號或換行分隔\n例如：Niacinamide, Hyaluronic Acid, Retinol",
                        lines=3
                    )
                    text_btn = gr.Button("分析", variant="primary")
                with gr.Column():
                    text_output = gr.Textbox(
                        label="分析結果",
                        lines=25,
                        show_copy_button=True,
                    )

            gr.Examples(
                examples=[
                    ["Niacinamide"],
                    ["Hyaluronic Acid, Ceramide NP, Glycerin"],
                    ["Retinol, Salicylic Acid"],
                    ["Fragrance, Sodium Lauryl Sulfate"],
                    ["Bakuchiol"],
                ],
                inputs=text_input,
            )

            text_btn.click(fn=analyze_text, inputs=text_input, outputs=text_output)
            text_input.submit(fn=analyze_text, inputs=text_input, outputs=text_output)

        with gr.Tab("📷 圖片上傳"):
            gr.Markdown(
                "上傳化妝品成分標籤圖片，系統會自動辨識成分列表並分析。\n\n"
                "> ⚠️ 請上傳清晰的成分表照片（支援 JPG、PNG、WEBP）"
            )
            with gr.Row():
                with gr.Column():
                    image_input = gr.File(
                        label="上傳成分標籤圖片",
                        file_types=[".png", ".jpg", ".jpeg", ".webp"],
                    )
                    image_btn = gr.Button("辨識並分析", variant="primary")
                with gr.Column():
                    image_output = gr.Textbox(
                        label="分析結果",
                        lines=25,
                        show_copy_button=True,
                    )

            image_btn.click(fn=analyze_image, inputs=image_input, outputs=image_output)

    gr.Markdown(
        "---\n"
        "**資料來源**：CosIng（歐盟官方）、INCI Decoder\n\n"
        "**confidence**：`high` = 知識庫　`medium` = AI 生成　`error` = 查詢失敗"
    )

if __name__ == "__main__":
    demo.launch(share=True)