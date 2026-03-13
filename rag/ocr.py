import os
import base64
import json
from groq import Groq

API_KEYS = [
    key for key in [
        os.environ.get("GROQ_API_KEY_1"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    if key
]

OCR_PROMPT = """
請辨識這張圖片中的化妝品成分列表（Ingredients）。

請嚴格按照以下 JSON 格式輸出，不要加任何額外說明：
{
  "ingredients_text": "完整的成分列表原文",
  "ingredients": ["成分1", "成分2", "成分3"]
}

注意：
- ingredients_text 是圖片中成分列表的完整原文
- ingredients 是拆解後的單一成分陣列
- 成分名稱保留原文（通常是英文 INCI 名稱）
- 如果圖片中沒有成分列表，ingredients_text 和 ingredients 都填空
- 忽略「Ingredients:」或「成分:」這樣的標題文字
"""


def _encode_image(image_path: str) -> tuple[str, str]:
    """
    將圖片檔案編碼為 base64。
    回傳 (base64字串, media_type)
    """
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, media_type


def extract_from_file(image_path: str) -> dict:
    """
    從圖片檔案辨識成分列表。

    Args:
        image_path: 圖片檔案路徑

    回傳：
    {
        "ingredients_text": "Water, Niacinamide, ...",
        "ingredients": ["Water", "Niacinamide", ...]
    }
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

    b64, media_type = _encode_image(image_path)
    return _call_vision(b64, media_type)


def extract_from_base64(b64_string: str, media_type: str = "image/jpeg") -> dict:
    """
    從 base64 字串辨識成分列表。
    用於 Gradio 介面直接傳入圖片資料的情況。

    Args:
        b64_string: base64 編碼的圖片字串
        media_type: 圖片格式，預設 image/jpeg

    回傳：
    {
        "ingredients_text": "Water, Niacinamide, ...",
        "ingredients": ["Water", "Niacinamide", ...]
    }
    """
    if not API_KEYS:
        raise ValueError("沒有可用的 GROQ_API_KEY，請設定環境變數")

    return _call_vision(b64_string, media_type)


def _call_vision(b64: str, media_type: str) -> dict:
    """呼叫 Groq Vision API，支援多 key 輪替"""
    last_error = None
    for api_key in API_KEYS:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT
                            }
                        ]
                    }
                ],
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方式：python rag/ocr.py <圖片路徑>")
        print("範例：python rag/ocr.py test_label.jpg")
        sys.exit(0)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"找不到圖片：{image_path}")
        sys.exit(1)

    print(f"辨識圖片：{image_path}")
    result = extract_from_file(image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))