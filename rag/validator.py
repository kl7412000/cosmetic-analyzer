import os
import json
from groq import Groq

# 必填欄位定義
REQUIRED_FIELDS = [
    "ingredient", "inci_name", "cas_number",
    "functions", "benefits", "risks",
    "skin_type", "eu_regulation", "source"
]

LIST_FIELDS = ["functions", "benefits", "risks", "skin_type", "source"]

API_KEYS = [
    key for key in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    if key
]

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
    """
    Layer 1：格式驗證。
    回傳 (是否通過, 錯誤訊息列表)
    """
    errors = []

    # 檢查必填欄位
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"缺少必填欄位：{field}")

    # 檢查 list 欄位型別
    for field in LIST_FIELDS:
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} 應為 list，實際為 {type(data[field]).__name__}")

    # 檢查非空字串欄位
    for field in ["ingredient", "inci_name", "eu_regulation"]:
        if field in data and isinstance(data[field], str) and not data[field].strip():
            errors.append(f"{field} 不可為空字串")

    passed = len(errors) == 0
    return passed, errors


def validate_quality(data: dict) -> dict:
    """
    Layer 3：LLM-as-a-judge 品質評估（離線流程使用）。
    回傳 {"score": int, "issues": list, "verdict": str}
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

    data_text = json.dumps(data, ensure_ascii=False, indent=2)
    prompt = JUDGE_PROMPT.format(data=data_text)

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

            result = json.loads(raw.strip())

            # 確保 verdict 和 score 一致
            score = result.get("score", 0)
            if score >= 7:
                result["verdict"] = "pass"
            elif score >= 4:
                result["verdict"] = "unverified"
            else:
                result["verdict"] = "fail"

            return result

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")


def validate(data: dict, run_judge: bool = True) -> dict:
    """
    完整驗證流程（格式驗證 + LLM-as-a-judge）。

    Args:
        data: 要驗證的成分資料
        run_judge: 是否執行 LLM-as-a-judge（離線流程傳 True，線上 fallback 傳 False）

    回傳：
    {
        "passed": bool,          # 是否通過格式驗證
        "format_errors": list,   # 格式錯誤列表
        "score": int,            # LLM 評分（0-10，run_judge=False 時為 -1）
        "issues": list,          # LLM 發現的問題
        "verdict": str,          # "pass" / "unverified" / "fail" / "format_error"
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

    # 線上 fallback：只做格式驗證，verdict 預設 medium
    return {
        "passed": True,
        "format_errors": [],
        "score": -1,
        "issues": [],
        "verdict": "pass",
    }


if __name__ == "__main__":
    # 測試 1：正常資料
    good_data = {
        "ingredient": "Niacinamide",
        "inci_name": "Niacinamide",
        "cas_number": "98-92-0",
        "functions": ["Skin conditioning", "Sebum control"],
        "benefits": ["提亮膚色", "縮小毛孔", "控油"],
        "risks": ["高濃度可能造成輕微泛紅"],
        "skin_type": ["油性肌", "混合肌"],
        "eu_regulation": "無使用限制",
        "source": ["CosIng", "LLM-generated"]
    }

    # 測試 2：格式錯誤的資料
    bad_format_data = {
        "ingredient": "Niacinamide",
        "inci_name": "",           # 空字串
        "functions": "Smoothing",  # 應為 list
        # 缺少多個必填欄位
    }

    # 測試 3：假造的成分（LLM 應該給低分）
    fake_data = {
        "ingredient": "MagicYouthSerum9000",
        "inci_name": "MAGICYOUTHSERUM9000",
        "cas_number": "000-00-0",
        "functions": ["ANTI-AGING"],
        "benefits": ["讓皮膚年輕20歲"],
        "risks": ["無任何風險"],
        "skin_type": ["所有膚質"],
        "eu_regulation": "無使用限制",
        "source": ["LLM-generated"]
    }

    print("=== 測試 1：正常資料（格式驗證 + LLM judge）===")
    result1 = validate(good_data, run_judge=True)
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    print("\n=== 測試 2：格式錯誤資料 ===")
    result2 = validate(bad_format_data, run_judge=False)
    print(json.dumps(result2, ensure_ascii=False, indent=2))

    print("\n=== 測試 3：假造成分（LLM judge 應給低分）===")
    result3 = validate(fake_data, run_judge=True)
    print(json.dumps(result3, ensure_ascii=False, indent=2))