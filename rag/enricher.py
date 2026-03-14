import os
import json
from groq import Groq

PROMPT_TEMPLATE = """
你是一個專業的化妝品成分分析師。
以下是從 CosIng 和 INCI Decoder 爬取的成分原始資料：

{scraped_data}

請根據你的專業知識，補充以下欄位，並嚴格按照 JSON 格式輸出，不要加任何額外說明：

{{
  "ingredient": "成分的常用英文名稱，例如 Niacinamide、Hyaluronic Acid",
  "inci_name": "{inci_name}",
  "cas_number": "{cas_number}",
  "functions": {functions},
  "benefits": ["消費者導向的功效說明，用繁體中文，3-5 條"],
  "risks": ["使用風險或注意事項，用繁體中文，1-3 條，如果非常安全可以只寫 1 條"],
  "skin_type": ["適合的膚質，用繁體中文，從以下選擇：油性肌、乾性肌、混合肌、敏感肌、痘痘肌、熟齡肌、暗沉肌、所有膚質"],
  "eu_regulation": "歐盟法規說明，用繁體中文，一句話說明限制或「無使用限制」",
  "source": {source}
}}

注意：
- benefits 要站在消費者角度，說明實際使用效果
- risks 要客觀，不要誇大，有科學依據
- skin_type 選最適合的 1-3 種，不要全選
- eu_regulation 參考 annex_refs 欄位，如果有濃度限制要寫清楚
- source 必須是陣列格式，例如 ["CosIng", "INCI Decoder"]
"""

API_KEYS = [
    key for key in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    if key
]


def _call_groq(prompt: str) -> dict:
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

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
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                raw = "\n".join(lines)
            return json.loads(raw.strip())
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")


def _normalize_source(source) -> list:
    """確保 source 是 list"""
    if isinstance(source, list):
        return source
    if isinstance(source, str):
        return [source]
    return []


def enrich(scraped: dict) -> dict:
    """
    接收爬蟲原始資料，用 LLM 補充缺少的欄位。
    """
    inci_name = scraped.get("inci_name", "") or scraped.get("name", "")
    cas_number = scraped.get("cas_number", "")
    functions = scraped.get("functions", [])
    source = _normalize_source(scraped.get("source", []))

    scraped_text = json.dumps(scraped, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(
        scraped_data=scraped_text,
        inci_name=inci_name,
        cas_number=cas_number,
        functions=json.dumps(functions, ensure_ascii=False),
        source=json.dumps(source, ensure_ascii=False),
    )

    result = _call_groq(prompt)

    # functions 去重（保留順序）
    if "functions" in result and isinstance(result["functions"], list):
        seen = set()
        result["functions"] = [
            f for f in result["functions"]
            if f.upper() not in seen and not seen.add(f.upper())
        ]
    
    # 確保 source 是 list
    result["source"] = _normalize_source(result.get("source", source))

    return result


def enrich_from_name(ingredient_name: str, confidence: str = "medium") -> dict:
    """
    只有成分名稱時直接用 LLM 生成完整資料。
    用於線上流程的 fallback。
    """
    prompt = f"""
你是一個專業的化妝品成分分析師。
請根據你的專業知識，為以下化妝品成分生成資料。

成分名稱：{ingredient_name}

請嚴格按照以下 JSON 格式輸出，不要加任何額外說明：
{{
  "ingredient": "成分的常用英文名稱",
  "inci_name": "INCI 標準名稱（英文大寫）",
  "cas_number": "CAS 號碼，不確定請填空字串",
  "functions": ["INCI 功能分類（英文大寫）"],
  "benefits": ["消費者導向的功效說明，用繁體中文，3-5 條"],
  "risks": ["使用風險或注意事項，用繁體中文，1-3 條"],
  "skin_type": ["適合的膚質，用繁體中文"],
  "eu_regulation": "歐盟法規說明，用繁體中文",
  "source": ["LLM-generated"],
  "confidence": "{confidence}",
  "warning": "此資料由 AI 生成，建議參考專業來源自行查證"
}}

如果你對這個成分不熟悉，請在 confidence 填 "low"。
"""
    result = _call_groq(prompt)
    
    # functions 去重（保留順序）
    if "functions" in result and isinstance(result["functions"], list):
        seen = set()
        result["functions"] = [
            f for f in result["functions"]
            if f.upper() not in seen and not seen.add(f.upper())
        ]

    # 確保必要欄位存在
    result["source"] = _normalize_source(result.get("source", ["LLM-generated"]))
    if "warning" not in result:
        result["warning"] = "此資料由 AI 生成，建議參考專業來源自行查證"

    return result


if __name__ == "__main__":
    test_scraped = {
        "source": "INCI Decoder",   # 字串，測試 normalize
        "name": "Bakuchiol",
        "inci_name": "",
        "cas_number": "10309-37-2",
        "functions": ["cell-communicating ingredient", "antioxidant"],
        "also_called": ["Sytenol A"],
    }

    print("=== 測試 enrich()（source 為字串）===")
    result = enrich(test_scraped)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nsource 型別：{type(result['source'])}，值：{result['source']}")