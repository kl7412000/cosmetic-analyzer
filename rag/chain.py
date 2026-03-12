import os
import json
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rag.retriever import load_retriever

PROMPT_TEMPLATE = """
你是一個專業的化妝品成分分析師。
請根據以下提供的成分資料，回答用戶的問題。
你只能根據提供的資料回答，不可以自己編造資訊。

成分資料：
{context}

用戶問題：
{query}

請嚴格按照以下 JSON 格式回答，不要加任何額外說明：
{{
  "ingredient": "成分名稱",
  "inci_name": "INCI 名稱",
  "cas_number": "CAS 號碼",
  "functions": ["功能1", "功能2"],
  "benefits": ["功效1", "功效2"],
  "risks": [],
  "skin_type": [],
  "eu_regulation": "歐盟法規說明",
  "source": ["來源1", "來源2"]
}}
"""

# 從環境變數讀取三個 key，過濾掉空值
API_KEYS = [
    key for key in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    if key
]

def build_chain(api_key: str):
    """
    建立 LangChain RAG chain
    input:  api_key（Groq API key）
    output: LangChain chain 物件
    """
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "query"]
    )

    chain = prompt | llm | JsonOutputParser()
    return chain

@traceable(name="analyze_ingredient")
def analyze(query: str) -> dict:
    """
    輪流嘗試三個 API key，額度用完自動換下一個
    input:  query（用戶輸入的成分名稱）
    output: 結構化的成分分析結果（dict）
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

    # 載入 retriever
    retriever = load_retriever(k=3)

    # 取得相關資料
    docs = retriever.invoke(query)

    # 把 retrieved documents 的 metadata 整理成 context
    context = json.dumps(
        [doc.metadata for doc in docs],
        ensure_ascii=False,
        indent=2
    )

    # 輪流嘗試每個 key
    last_error = None
    for api_key in API_KEYS:
        try:
            chain = build_chain(api_key)
            result = chain.invoke({
                "context": context,
                "query": query
            })
            return result

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")


def parse_ingredients(text: str) -> list:
    """
    把用戶輸入的多成分文字拆成 list
    支援逗號、換行、頓號分隔
    input:  "Niacinamide, Hyaluronic Acid, Retinol"
    output: ["Niacinamide", "Hyaluronic Acid", "Retinol"]
    """
    import re
    # 支援逗號、換行、頓號、分號分隔
    ingredients = re.split(r'[,，、;\n]+', text)
    # 清除空白、過濾空字串
    ingredients = [i.strip() for i in ingredients if i.strip()]
    return ingredients


@traceable(name="analyze_multiple_ingredients")
def analyze_multiple(text: str) -> list:
    """
    分析多個成分，回傳每個成分的分析結果
    input:  用戶輸入的成分文字（可多個）
    output: 每個成分的分析結果 list
    """
    ingredients = parse_ingredients(text)

    if not ingredients:
        raise ValueError("請輸入至少一個成分名稱")

    results = []
    for ingredient in ingredients:
        try:
            result = analyze(ingredient)
            results.append({
                "status": "success",
                "data": result
            })
        except Exception as e:
            results.append({
                "status": "error",
                "ingredient": ingredient,
                "message": str(e)
            })

    return results