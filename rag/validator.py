# rag/validator.py
import json
from rag.groq_client import call_groq   # ← 改用共用模組

# 必填欄位定義
REQUIRED_FIELDS = [
    "ingredient", "inci_name", "cas_number",
    "functions", "benefits", "risks",
    "skin_type", "eu_regulation", "source"
]

LIST_FIELDS = ["functions", "benefits", "risks", "skin_type", "source"]

JUDGE_PROMPT = """
你是一個化妝品成分資料的品質審查員。
請評估以下成分資料的準確性和完整性。

成分資料：
{data}

請嚴格按照以下 JSON 格式回答，不要加任何額外說明：
{{
  "score": 評分（1-10 的整數，10 分最高）,
  "issues": ["發現的問題1", "發現的問題2"],
  "verdict": "pass" 或 "unverified" 或 "fail"
}}

評分標準：
- 9-10：資料準確、完整，無明顯問題
- 7-8：資料大致正確，有小瑕疵但不影響使用
- 4-6：資料有部分不確定或輕微錯誤
- 1-3：資料明顯錯誤或嚴重不完整

verdict 規則：
- score >= 7 → "pass"
- score 4-6 → "unverified"
- score <= 3 → "fail"

特別注意：
- benefits 和 risks 是否符合該成分的已知特性
- eu_regulation 是否合理
- skin_type 是否適當
- 如果成分不存在或名稱明顯錯誤，score 給 1-2
"""


def validate_format(data: dict) -> tuple[bool, list[str]]:
    """Layer 1：格式驗證。回傳 (是否通過, 錯誤訊息列表)"""
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"缺少必填欄位：{field}")

    for field in LIST_FIELDS:
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} 應為 list，實際為 {type(data[field]).__name__}")

    for field in ["ingredient", "inci_name", "eu_regulation"]:
        if field in data and isinstance(data[field], str) and not data[field].strip():
            errors.append(f"{field} 不可為空字串")

    passed = len(errors) == 0
    return passed, errors


def validate_quality(data: dict) -> dict:
    """Layer 3：LLM-as-a-judge 品質評估（離線流程使用）。"""
    data_text = json.dumps(data, ensure_ascii=False, indent=2)
    prompt = JUDGE_PROMPT.format(data=data_text)

    result = call_groq(prompt)   # ← 替換（rate limit 重試已在 groq_client 處理）

    score = result.get("score", 0)
    if score >= 7:
        result["verdict"] = "pass"
    elif score >= 4:
        result["verdict"] = "unverified"
    else:
        result["verdict"] = "fail"

    return result


def validate(data: dict, run_judge: bool = True) -> dict:
    """
    完整驗證流程（格式驗證 + LLM-as-a-judge）。

    回傳：
    {
        "passed": bool,
        "format_errors": list,
        "score": int,
        "issues": list,
        "verdict": str,   # "pass" / "unverified" / "fail" / "format_error"
    }
    """
    # Layer 1：格式驗證
    passed, format_errors = validate_format(data)

    if not passed:
        return {
            "passed": False,
            "format_errors": format_errors,
            "score": -1,
            "issues": [],
            "verdict": "format_error",
        }

    # Layer 3：LLM-as-a-judge（只在離線流程執行）
    if run_judge:
        judge_result = validate_quality(data)
        return {
            "passed": True,
            "format_errors": [],
            "score": judge_result.get("score", -1),
            "issues": judge_result.get("issues", []),
            "verdict": judge_result.get("verdict", "unverified"),
        }

    return {
        "passed": True,
        "format_errors": [],
        "score": -1,
        "issues": [],
        "verdict": "pass",
    }


if __name__ == "__main__":
    good_data = {
        "ingredient": "Niacinamide",
        "inci_name": "Niacinamide",
        "cas_number": "98-92-0",
        "functions": ["skin conditioning", "sebum control"],
        "benefits": ["brightens skin tone", "minimizes pores", "controls oil"],
        "risks": ["may cause flushing at high concentrations"],
        "skin_type": ["oily skin", "combination skin"],
        "eu_regulation": "No restrictions",
        "source": ["CosIng", "LLM-generated"]
    }

    print("=== 測試：正常資料（格式驗證 + LLM judge）===")
    result = validate(good_data, run_judge=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))