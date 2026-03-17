import os
import json
import base64
import io
import tempfile
import gradio as gr
import gradio.blocks
from PIL import Image
from rag.graph import analyze_online


# 修復 gradio bug
gradio.blocks.Blocks.get_api_info = lambda self: {"named_endpoints": {}, "unnamed_endpoints": {}}


# ==============================
# 文青風 CSS
# ==============================

custom_css = """

body, .gradio-container{
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang TC","Noto Sans TC",sans-serif;
}

/* LIGHT MODE */

body{
background:#f7f7f5;
color:#2c2c2c;
}

.ingredient-card{
background:#ffffff;
border:1px solid #e8e8e8;
border-left:6px solid #a3a3a3;
border-radius:12px;
padding:20px;
margin-bottom:16px;
box-shadow:0 3px 8px rgba(0,0,0,0.05);
transition:all 0.2s;
}

.ingredient-card:hover{
transform:translateY(-2px);
box-shadow:0 6px 14px rgba(0,0,0,0.08);
}

.card-high{
border-left-color:#63b38b;
}

.card-medium{
border-left-color:#e3a54a;
}

.card-error{
border-left-color:#d9534f;
}

.card-title{
font-size:18px;
font-weight:600;
margin-bottom:4px;
}

.card-sub{
font-size:13px;
opacity:0.7;
margin-bottom:8px;
}

details summary{
cursor:pointer;
margin-top:8px;
font-weight:500;
}

.tag-high{
background:#e6f4ea;
color:#2e7d32;
padding:3px 8px;
border-radius:6px;
font-size:12px;
margin-left:6px;
}

.tag-medium{
background:#fff4e5;
color:#b26a00;
padding:3px 8px;
border-radius:6px;
font-size:12px;
margin-left:6px;
}


/* DARK MODE */

@media (prefers-color-scheme: dark){

body{
background:#1f2127;
color:#e4e4e4;
}

.ingredient-card{
background:#2a2d35;
border:1px solid #3a3f47;
}

.tag-high{
background:#1e3d2f;
color:#7ee2a8;
}

.tag-medium{
background:#3d2e17;
color:#ffc76a;
}

}

"""


# ==============================
# session
# ==============================

session_pending = []


# ==============================
# 卡片顯示
# ==============================

def format_card(item:dict):

    name = item.get("_display_name") or item.get("_original") or item.get("ingredient") or item.get("inci_name", "Unknown")

    confidence = item.get("confidence","medium").lower()

    tag_class = "tag-high" if confidence=="high" else "tag-medium"

    tag_text = "官方資料" if confidence=="high" else "AI 推論"

    card_class=f"ingredient-card card-{confidence}"

    inci=item.get("inci_name","")
    cas=item.get("cas_number","")
    functions=item.get("functions",[])
    benefits=item.get("benefits",[])
    risks=item.get("risks",[])
    eu_reg=item.get("eu_regulation","")
    skin=item.get("skin_type",[])
    warning=item.get("warning","")

    md=f"""
    <div class="{card_class}">

    <div class="card-title">
    {name}
    <span class="{tag_class}">{tag_text}</span>
    </div>

    <div class="card-sub">
    {inci if inci else ""} {(" | CAS "+cas) if cas else ""}
    </div>

    **功能：** {", ".join(functions) if functions else "—"}

    <details>
    <summary>查看詳細</summary>

    """

    if benefits:
        md+="\n**功效**\n"
        for b in benefits:
            md+=f"- {b}\n"

    if risks:
        md+="\n**風險**\n"
        for r in risks:
            md+=f"- ⚠️ {r}\n"

    if skin:
        md+=f"\n**適合膚質：** {', '.join(skin)}\n"

    if eu_reg:
        md+=f"\n**EU 法規：** {eu_reg}\n"

    if warning:
        md+=f"\n> ⚠️ {warning}\n"

    md+="</details></div>"

    return md


# ==============================
# 結果整理
# ==============================

def process_results(results:list):

    if not results:
        return "### ⚠️ 未找到任何成分"

    output=""

    for r in results:

        if r.get("error") and r.get("confidence")=="error":

            output+=f"""
<div class="ingredient-card card-error">
<div class="card-title">{r.get('ingredient','unknown')}</div>
查詢失敗
</div>
"""

        else:

            output+=format_card(r)

    return output


# ==============================
# 文字分析
# ==============================

def analyze_text(ingredient_input:str):

    if not ingredient_input.strip():
        return "### 💡 請輸入成分名稱",gr.DownloadButton(visible=False)

    try:

        results=analyze_online(ingredient_input.strip())

        for r in results:

            if r.get("confidence")=="medium" and r.get("source")==["LLM-generated"]:

                clean={k:v for k,v in r.items() if k not in("_query","_original","is_substance")}

                session_pending.append(clean)

        if session_pending:

            tmp=tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False,encoding="utf-8")

            json.dump(session_pending,tmp,ensure_ascii=False,indent=2)

            tmp.close()

            download_btn=gr.DownloadButton(visible=True,value=tmp.name)

        else:

            download_btn=gr.DownloadButton(visible=False)

        return process_results(results),download_btn

    except Exception as e:

        return f"### ❌ 發生錯誤\n`{str(e)}`",gr.DownloadButton(visible=False)


# ==============================
# 圖片分析
# ==============================

def analyze_image(files):

    if not files:
        return "### 💡 請上傳圖片"

    try:

        file=files[0]

        image=Image.open(file.name)

        buffered=io.BytesIO()

        image.save(buffered,format="JPEG")

        b64=base64.b64encode(buffered.getvalue()).decode()

        results=analyze_online("",image_b64=b64)

        return process_results(results)

    except Exception as e:

        return f"### ❌ 圖片處理失敗\n`{str(e)}`"


# ==============================
# UI
# ==============================

with gr.Blocks(
title="Cosmetic Ingredient Analyzer",
css=custom_css,
theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# 🧴 Cosmetic Ingredient Analyzer")

    gr.Markdown("CosIng + RAG 成分分析系統")

    with gr.Tabs():

        with gr.Tab("文字分析"):

            with gr.Row():

                with gr.Column(scale=1):

                    text_input=gr.Textbox(
                        label="輸入成分",
                        placeholder="Niacinamide, Retinol, Glycerin",
                        lines=4
                    )

                    text_btn=gr.Button("開始分析",variant="primary")

                    download_btn=gr.DownloadButton(
                        label="下載 AI 成分資料",
                        visible=False
                    )

                with gr.Column(scale=2):

                    text_output=gr.Markdown(value="等待輸入...")

            gr.Examples(
                examples=[
                    ["Salicylic Acid, Glycerin"],
                    ["Niacinamide, Hyaluronic Acid"],
                    ["Retinol"],
                    ["Bakuchiol"],
                ],
                inputs=text_input
            )

        with gr.Tab("圖片辨識"):

            with gr.Row():

                with gr.Column(scale=1):

                    image_input=gr.File(
                        label="上傳成分表",
                        file_types=["image"],
                        file_count="multiple"
                    )

                    image_btn=gr.Button("辨識並分析",variant="primary")

                with gr.Column(scale=2):

                    image_output=gr.Markdown(value="請上傳圖片")

    gr.Markdown(
"""
---
資料來源：CosIng / INCI Decoder  
AI 生成資料僅供參考
"""
)

    text_btn.click(fn=analyze_text,inputs=text_input,outputs=[text_output,download_btn])
    text_input.submit(fn=analyze_text,inputs=text_input,outputs=[text_output,download_btn])
    image_btn.click(fn=analyze_image,inputs=image_input,outputs=image_output)


if __name__=="__main__":

    if not os.path.exists("faiss_index/index.faiss"):
        raise FileNotFoundError("找不到 FAISS 索引")

    is_hf=os.environ.get("SPACE_ID") is not None

    demo.launch(
        server_name="0.0.0.0" if is_hf else "127.0.0.1"
    )