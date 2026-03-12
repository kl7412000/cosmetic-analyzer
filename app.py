import json
import os
import gradio as gr
from rag.chain import analyze

# 確認索引檔案存在
if not os.path.exists("faiss_index/index.faiss"):
    raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")

def analyze_ingredient(ingredient_name: str) -> str:
    if not ingredient_name.strip():
        return "請輸入成分名稱"

    try:
        result = analyze(ingredient_name)
        return json.dumps(result, ensure_ascii=False, indent=2)

    except RuntimeError as e:
        return f"API 錯誤：{str(e)}"
    except Exception as e:
        return f"發生錯誤：{str(e)}"


with gr.Blocks(title="Cosmetic Ingredient Analyzer") as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown("輸入化妝品成分名稱，獲取功效、法規資訊與來源說明")

    with gr.Row():
        with gr.Column():
            ingredient_input = gr.Textbox(
                label="成分名稱",
                placeholder="例如：Niacinamide、Hyaluronic Acid、Retinol",
                lines=1
            )
            submit_btn = gr.Button("分析", variant="primary")

        with gr.Column():
            output = gr.Code(
                label="分析結果",
                language="json",
                lines=20
            )

    gr.Examples(
        examples=[
            ["Niacinamide"],
            ["Hyaluronic Acid"],
            ["Retinol"],
            ["Salicylic Acid"],
            ["Fragrance"],
        ],
        inputs=ingredient_input,
    )

    submit_btn.click(fn=analyze_ingredient, inputs=ingredient_input, outputs=output)
    ingredient_input.submit(fn=analyze_ingredient, inputs=ingredient_input, outputs=output)

if __name__ == "__main__":
    demo.launch()