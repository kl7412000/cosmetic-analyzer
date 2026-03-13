import os
import json
import base64
import gradio as gr
from rag.graph import analyze_online

if not os.path.exists("faiss_index/index.faiss"):
    raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")


def format_result(item: dict) -> str:
    """將單一成分結果格式化為易讀的 JSON 字串"""
    # 移除內部使用的欄位
    display = {k: v for k, v in item.items()
               if k not in ("_query", "is_substance")}
    return json.dumps(display, ensure_ascii=False, indent=2)


def analyze_text(ingredient_input: str) -> str:
    """處理文字輸入"""
    if not ingredient_input.strip():
        return "請輸入成分名稱"

    try:
        results = analyze_online(ingredient_input.strip())
        if not results:
            return "無結果"
        if len(results) == 1:
            return format_result(results[0])
        # 多個成分
        output = []
        for r in results:
            name = r.get("ingredient") or r.get("inci_name", "unknown")
            output.append(f"// ── {name} ──")
            output.append(format_result(r))
        return "\n\n".join(output)

    except Exception as e:
        return f"發生錯誤：{e}"


def analyze_image(image) -> str:
    """處理圖片輸入"""
    if image is None:
        return "請上傳圖片"

    try:
        # Gradio 回傳 numpy array，先轉成 PIL 再轉 base64
        import numpy as np
        from PIL import Image
        import io

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype("uint8"))
        else:
            pil_image = image

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

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


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Cosmetic Ingredient Analyzer") as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown(
        "輸入化妝品成分名稱或上傳成分標籤圖片，獲取功效、風險、法規資訊。\n\n"
        "- **已知成分**：直接從知識庫查詢（高可信度）\n"
        "- **未知成分**：由 AI 即時生成（中等可信度，附警告）"
    )

    with gr.Tabs():

        # ── Tab 1：文字輸入 ──────────────────────────────────────────────────
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
                    text_output = gr.Textbox(label="分析結果", lines=25)

            gr.Examples(
                examples=[
                    ["Niacinamide"],
                    ["Hyaluronic Acid, Ceramide NP, Glycerin"],
                    ["Retinol, Salicylic Acid"],
                    ["Fragrance, Sodium Lauryl Sulfate"],
                    ["Bakuchiol"],  # 未知成分，走 LLM fallback
                ],
                inputs=text_input,
            )

            text_btn.click(fn=analyze_text, inputs=text_input, outputs=text_output)
            text_input.submit(fn=analyze_text, inputs=text_input, outputs=text_output)

        # ── Tab 2：圖片上傳 ──────────────────────────────────────────────────
        with gr.Tab("📷 圖片上傳"):
            gr.Markdown(
                "上傳化妝品成分標籤圖片，系統會自動辨識成分列表並分析。\n\n"
                "> ⚠️ 圖片辨識需要清晰的成分列表，建議拍攝產品背面成分表。"
            )
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="上傳成分標籤圖片",
                        type="numpy",
                    )
                    image_btn = gr.Button("辨識並分析", variant="primary")

                with gr.Column():
                    image_output = gr.Code(
                        label="分析結果",
                        language="json",
                        lines=25
                    )

            image_btn.click(fn=analyze_image, inputs=image_input, outputs=image_output)

    gr.Markdown(
        "---\n"
        "**資料來源**：CosIng（歐盟官方）、INCI Decoder\n\n"
        "**confidence 說明**：`high` = 知識庫資料　`medium` = AI 生成，附警告　`error` = 查詢失敗"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")