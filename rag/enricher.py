# rag/enricher.py
import json
from rag.groq_client import call_groq   # ← 改用共用模組

# ── Prompt 設計 ────────────────────────────────────────────────────────
ENRICH_PROMPT = """
你是一個專業的化妝品成分分析師，熟悉 INCI 命名規則與歐盟化妝品法規。

{context}

請嚴格按照以下 JSON 格式回答，不要加任何額外說明或 markdown：
{{
  "ingredient": "成分英文常用名稱",
  "inci_name": "INCI 官方英文名稱",
  "cas_number": "CAS 號碼，可以有多個用 / 分隔，不知道就填 N/A",
  "functions": ["function1", "function2"],
  "benefits": ["benefit1", "benefit2"],
  "risks": ["risk1（若無請填空 list）"],
  "skin_type": ["skin type1", "skin type2（若無法判斷請填空 list）"],
  "eu_regulation": "EU regulation description in English",
  "source": ["source1"]
}}

語言規則（非常重要）：
- ingredient 和 inci_name 必須是英文
- functions 必須是英文，使用小寫，參考 INCI Decoder 的分類詞
  例如：antioxidant, emollient, humectant, preservative, surfactant/cleansing,
        skin-identical ingredient, cell-communicating ingredient, soothing,
        exfoliant, uv filter, solvent, antimicrobial/antibacterial
- benefits 必須是英文，每筆簡短描述（10 字以內）
  例如：brightens skin tone, reduces fine lines, hydrates skin
- risks 必須是英文，每筆簡短描述
  例如：may cause irritation, not recommended for sensitive skin
- skin_type 必須是英文
  例如：all skin types, oily skin, dry skin, sensitive skin, acne-prone skin
- eu_regulation 必須是英文
  例如：No restrictions, Maximum concentration 0.3% in face products

格式規則：
- 每個 list 欄位至少填一筆，不要留空 list（skin_type 和 risks 例外）
- benefits 至少填 2 筆，最多 5 筆
- functions 至少填 1 筆，最多 4 筆
"""

ENRICH_CONTEXT = """
以下是我從網路上爬取的部分成分資料，請根據這些資訊補充缺少的欄位（benefits、risks、skin_type）：

{scraped_json}

請根據你的專業知識補充上述資料缺少的欄位，並輸出完整的 JSON。
"""

FROM_NAME_CONTEXT = """
請根據你的專業知識，為以下化妝品成分生成完整的分析資料：

成分名稱：{ingredient_name}

注意：只能填入你確實知道的資訊，不確定的欄位寫 N/A 或空 list，不要編造。
"""


def _normalize_source(source) -> list:
    if isinstance(source, list):
        return source
    if isinstance(source, str):
        return [source]
    return []


def enrich(scraped: dict) -> dict:
    """
    離線流程使用。
    接收爬蟲已取得的部分資料，用 LLM 補充缺少的欄位。
    """
    context = ENRICH_CONTEXT.format(
        scraped_json=json.dumps(scraped, ensure_ascii=False, indent=2)
    )
    prompt = ENRICH_PROMPT.format(context=context)
    result = call_groq(prompt)   # ← 替換

    result["source"] = _normalize_source(result.get("source", []))

    scraped_source = _normalize_source(scraped.get("source", []))
    for s in scraped_source:
        if s not in result["source"]:
            result["source"].append(s)

    return result


def enrich_from_name(ingredient_name: str) -> dict:
    """
    線上流程使用（LLM fallback）。
    只有成分名稱時，從零生成完整的成分資料。
    """
    context = FROM_NAME_CONTEXT.format(ingredient_name=ingredient_name)
    prompt = ENRICH_PROMPT.format(context=context)
    result = call_groq(prompt)   # ← 替換

    result["source"] = ["LLM-generated"]
    result["confidence"] = "medium"
    result["warning"] = "此資料由 AI 生成，建議參考 INCI Decoder 或 Paula's Choice 等專業來源自行查證"

    return result


if __name__ == "__main__":
    print("=== 測試 enrich_from_name（線上 fallback）===")
    result = enrich_from_name("Bakuchiol")
    print(json.dumps(result, ensure_ascii=False, indent=2))