import json
import os
from groq import Groq

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

def generate(query: str, context: list) -> dict:
    """
    輪流嘗試三個 API key，額度用完自動換下一個
    input:  query（用戶問題）、context（retriever 找到的相關資料）
    output: 結構化的成分分析結果（dict）
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請在 Hugging Face Secrets 設定至少一個")

    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(context=context_text, query=query)

    # 輪流嘗試每個 key
    last_error = None
    for api_key in API_KEYS:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()

            # 清除可能的 markdown 格式
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except Exception as e:
            # 這個 key 失敗（額度用完或無效），換下一個
            last_error = e
            continue

    # 三個 key 都失敗
    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")
'''

---

## 在 Hugging Face Secrets 設定方式

之後部署到 Hugging Face 的時候，進到你的 Space 設定頁面，在 Secrets 區塊加入：

GROQ_API_KEY_1 = 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
GROQ_API_KEY_2 = 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
GROQ_API_KEY_3 = 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
by OCR_Receipt_251029_key
'''