import json
import gradio as gr
from rag.retriever import search
from rag.generator import generate

# 確認索引檔案存在
import os
if not os.path.exists("faiss_index/index.faiss"):
    raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")

def analyze(ingredient_name: str) -> str:
    """
    主要分析函數，串接 retriever 和 generator
    input:  成分名稱（用戶輸入）
    output: JSON 格式的分析結果（字串）
    """
    if not ingredient_name.strip():
        return "請輸入成分名稱"

    try:
        # Step 1：從 FAISS 找最相關的 3 筆資料
        context = search(ingredient_name, k=3)

        # Step 2：呼叫 Groq 生成結構化回答
        result = generate(ingredient_name, context)

        # Step 3：格式化輸出
        return json.dumps(result, ensure_ascii=False, indent=2)

    except RuntimeError as e:
        return f"API 錯誤：{str(e)}"
    except json.JSONDecodeError:
        return "解析回答失敗，請再試一次"
    except Exception as e:
        return f"發生錯誤：{str(e)}"


# 建立 Gradio 介面
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

    # 範例輸入
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

    submit_btn.click(
        fn=analyze,
        inputs=ingredient_input,
        outputs=output
    )

    ingredient_input.submit(
        fn=analyze,
        inputs=ingredient_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()