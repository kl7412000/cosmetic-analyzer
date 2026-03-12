import json
import os
import gradio as gr
from rag.chain import analyze_multiple

if not os.path.exists("faiss_index/index.faiss"):
    raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")


def analyze_ingredient(ingredient_text: str) -> str:
    if not ingredient_text.strip():
        return "請輸入成分名稱"

    try:
        results = analyze_multiple(ingredient_text)
        return json.dumps(results, ensure_ascii=False, indent=2)

    except ValueError as e:
        return f"輸入錯誤：{str(e)}"
    except RuntimeError as e:
        return f"API 錯誤：{str(e)}"
    except Exception as e:
        return f"發生錯誤：{str(e)}"


with gr.Blocks(title="Cosmetic Ingredient Analyzer") as demo:
    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")
    gr.Markdown("輸入化妝品成分名稱，支援多成分分析（用逗號或換行分隔）")

    with gr.Row():
        with gr.Column():
            ingredient_input = gr.Textbox(
                label="成分名稱",
                placeholder="單一成分：Niacinamide\n多個成分：Niacinamide, Hyaluronic Acid, Retinol",
                lines=3
            )
            submit_btn = gr.Button("分析", variant="primary")

        with gr.Column():
            output = gr.Code(
                label="分析結果",
                language="json",
                lines=25
            )

    gr.Examples(
        examples=[
            ["Niacinamide"],
            ["Hyaluronic Acid"],
            ["Niacinamide, Hyaluronic Acid"],
            ["Retinol, Salicylic Acid, Fragrance"],
        ],
        inputs=ingredient_input,
    )

    submit_btn.click(
        fn=analyze_ingredient,
        inputs=ingredient_input,
        outputs=output
    )
    ingredient_input.submit(
        fn=analyze_ingredient,
        inputs=ingredient_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()