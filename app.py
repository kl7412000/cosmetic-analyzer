import os
import json
import gradio as gr
from rag.graph import analyze_online

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


with gr.Blocks(title="Cosmetic Ingredient Analyzer") as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown(
        "輸入化妝品成分名稱，獲取功效、風險、歐盟法規資訊。\n\n"
        "- **已知成分**：從知識庫查詢（`confidence: high`）\n"
        "- **未知成分**：由 AI 即時生成（`confidence: medium`，附警告）"
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="成分名稱",
                placeholder="輸入一個或多個成分，用逗號或換行分隔\n例如：Niacinamide, Hyaluronic Acid, Retinol",
                lines=3
            )
            submit_btn = gr.Button("分析", variant="primary")

        with gr.Column():
            output = gr.Textbox(
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

    gr.Markdown(
        "---\n"
        "**資料來源**：CosIng（歐盟官方）、INCI Decoder、Paula's Choice\n\n"
        "**confidence**：`high` = 知識庫　`medium` = AI 生成　`error` = 查詢失敗"
    )

    submit_btn.click(fn=analyze_text, inputs=text_input, outputs=output)
    text_input.submit(fn=analyze_text, inputs=text_input, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")