# rag/enricher.py
import os
import json
from groq import Groq

# ── Groq API Key 輪替邏輯（與 chain.py 一致）──────────────────────────
API_KEYS = [
    key for key in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    if key
]

# ── Prompt 設計 ────────────────────────────────────────────────────────
# 這個 prompt 有兩種模式：
# 1. 有 scraped_data → 補充缺少的欄位（離線流程）
# 2. 沒有 scraped_data → 從零生成完整資料（線上流程 fallback）
ENRICH_PROMPT = """
你是一個專業的化妝品成分分析師，熟悉 INCI 命名規則與歐盟化妝品法規。

{context}

請嚴格按照以下 JSON 格式回答，不要加任何額外說明或 markdown：
{{
  "ingredient": "成分常用名稱",
  "inci_name": "INCI 官方名稱",
  "cas_number": "CAS 號碼，不知道就填 N/A",
  "functions": ["功能1", "功能2"],
  "benefits": ["功效1", "功效2"],
  "risks": ["風險1（若無請填空 list）"],
  "skin_type": ["適用膚質（若無法判斷請填空 list）"],
  "eu_regulation": "歐盟法規說明，不知道就填 無明確限制資訊",
  "source": ["來源1"]
}}
...
注意：
- ingredient 欄位必須使用英文名稱
- inci_name 欄位必須使用 INCI 官方英文名稱
- functions 欄位必須使用英文
...
"""

# 離線情境的 context 模板（有爬蟲資料可以參考）
ENRICH_CONTEXT = """
以下是我從網路上爬取的部分成分資料，請根據這些資訊補充缺少的欄位（benefits、risks、skin_type）：

{scraped_json}

請根據你的專業知識補充上述資料缺少的欄位，並輸出完整的 JSON。
"""

# 線上情境的 context 模板（從零生成）
FROM_NAME_CONTEXT = """
請根據你的專業知識，為以下化妝品成分生成完整的分析資料：

成分名稱：{ingredient_name}

注意：只能填入你確實知道的資訊，不確定的欄位寫 N/A 或空 list，不要編造。
"""


def _normalize_source(source) -> list:
    """
    統一 source 欄位的格式。
    INCI Decoder 爬蟲回傳字串 "INCI Decoder"，
    但我們的 JSON schema 要求 source 是 list。
    這個 helper 處理所有可能的輸入型別。
    """
    if isinstance(source, list):
        return source
    if isinstance(source, str):
        return [source]
    return []


def _call_groq(prompt: str) -> dict:
    """
    核心 LLM 呼叫函式，帶 API key 輪替邏輯。
    回傳解析後的 dict，或在所有 key 都失敗時拋出 RuntimeError。
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

    last_error = None
    for api_key in API_KEYS:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低 temperature 確保輸出穩定
            )
            raw = response.choices[0].message.content.strip()

            # 清理 LLM 可能回傳的 markdown code block
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            # JSON 解析失敗，這個 key 的輸出無效，嘗試下一個
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")


def enrich(scraped: dict) -> dict:
    """
    離線流程使用。
    接收爬蟲已取得的部分資料，用 LLM 補充缺少的欄位。

    參數：
        scraped: 爬蟲回傳的 dict，可能缺少 benefits/risks/skin_type

    回傳：
        補充完整的 dict，source 格式已正規化為 list
    """
    context = ENRICH_CONTEXT.format(
        scraped_json=json.dumps(scraped, ensure_ascii=False, indent=2)
    )
    prompt = ENRICH_PROMPT.format(context=context)
    result = _call_groq(prompt)

    # 確保 source 是 list（INCI Decoder 爬蟲可能回傳字串）
    result["source"] = _normalize_source(result.get("source", []))

    # 保留爬蟲來源，不讓 LLM 覆蓋掉
    scraped_source = _normalize_source(scraped.get("source", []))
    for s in scraped_source:
        if s not in result["source"]:
            result["source"].append(s)

    return result


def enrich_from_name(ingredient_name: str) -> dict:
    """
    線上流程使用（LLM fallback）。
    只有成分名稱時，從零生成完整的成分資料。
    自動標注 confidence: medium 和警告訊息。

    參數：
        ingredient_name: 成分常用名稱或 INCI 名稱

    回傳：
        完整的 dict，附帶 confidence 和 warning 欄位
    """
    context = FROM_NAME_CONTEXT.format(ingredient_name=ingredient_name)
    prompt = ENRICH_PROMPT.format(context=context)
    result = _call_groq(prompt)

    # 標注這筆資料是 LLM 生成的，讓前端可以顯示警告
    result["source"] = ["LLM-generated"]
    result["confidence"] = "medium"
    result["warning"] = "此資料由 AI 生成，建議參考 INCI Decoder 或 Paula's Choice 等專業來源自行查證"

    return result


# ── 本地測試 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 測試 enrich_from_name（線上 fallback）===")
    result = enrich_from_name("Bakuchiol")
    print(json.dumps(result, ensure_ascii=False, indent=2))