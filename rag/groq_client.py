# rag/groq_client.py
"""
共用 Groq API 呼叫模組。
- 多個 API key 輪替（不同帳號各自有獨立 TPD）
- 區分兩種 rate limit：
    TPD（tokens per day，每日配額耗盡）→ 直接換下一個 key
    RPM/TPM（每分鐘限制，暫時性）     → 等待後在同一個 key 重試
- 其他錯誤 → 直接換下一個 key
"""
import os
import re
import json
import time
import logging

from groq import Groq

logger = logging.getLogger(__name__)

# ── API Key 池 ────────────────────────────────────────────────────────
# 從環境變數讀取，支援無限個 key（KEY_1, KEY_2, KEY_3, ...）
def _load_api_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    # 也支援舊版單一 key
    fallback = os.environ.get("GROQ_API_KEY")
    if fallback and fallback not in keys:
        keys.append(fallback)
    return keys

API_KEYS: list[str] = _load_api_keys()

# ── 常數 ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES_PER_KEY = 3       # 每個 key 最多重試幾次（遇到 rate limit）
MAX_WAIT_SECONDS = 600        # 單次等待上限（10 分鐘）
BACKOFF_MULTIPLIER = 1.2      # 等待時間乘數（避免精準等待仍失敗）


def _parse_wait_seconds(error_message: str) -> float:
    """
    從 Groq 的 rate limit 錯誤訊息解析建議等待秒數。

    錯誤訊息格式範例：
      "Please try again in 5m42.144s."
      "Please try again in 30s."
      "Please try again in 1m."
    """
    # 優先解析 "XmYs" 格式
    match = re.search(r'in\s+(?:(\d+)m)?(?:([\d.]+)s)?', error_message)
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        total = minutes * 60 + seconds
        if total > 0:
            return total
    return 60.0  # 解析失敗時預設等 60 秒


def _is_rate_limit_error(exception: Exception) -> bool:
    """判斷是否為任何 rate limit（429）錯誤。"""
    msg = str(exception).lower()
    return "rate_limit_exceeded" in msg or "rate limit" in msg or "429" in msg


def _is_tpd_exhausted(exception: Exception) -> bool:
    """
    判斷是否為 TPD（tokens per day）每日配額耗盡。
    這種情況換 key 才有意義，等待沒有用。

    Groq 錯誤訊息特徵：
      'tokens per day (TPD): Limit 100000, Used 99922'
    """
    msg = str(exception)
    return "tokens per day" in msg.lower() or "TPD" in msg


def call_groq(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    呼叫 Groq API，帶多 key 輪替 + rate limit 智慧處理。

    rate limit 分兩種：
      TPD 耗盡（每日配額）→ 直接換下一個 key（等待沒用）
      RPM/TPM（每分鐘限制）→ 解析等待時間 → sleep → 同一個 key 重試

    回傳：
        解析後的 dict

    拋出：
        ValueError: 沒有設定任何 API key
        RuntimeError: 所有 key 全部失敗
    """
    if not API_KEYS:
        raise ValueError(
            "沒有可用的 GROQ_API_KEY，請設定環境變數 GROQ_API_KEY_1（或 GROQ_API_KEY）"
        )

    last_error = None

    for key_index, api_key in enumerate(API_KEYS):
        client = Groq(api_key=api_key)
        key_label = f"key_{key_index + 1}"

        for attempt in range(1, MAX_RETRIES_PER_KEY + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
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
                # JSON 解析失敗 → 換下一個 key
                logger.warning(f"[{key_label}] JSON 解析失敗：{e}")
                last_error = e
                break

            except Exception as e:
                last_error = e

                if _is_tpd_exhausted(e):
                    # TPD 耗盡：這個帳號今天沒額度了，直接換 key
                    print(f"  [{key_label}] TPD 耗盡，換下一個帳號")
                    break

                elif _is_rate_limit_error(e):
                    # RPM/TPM：暫時限速，等一下同一個 key 還可以用
                    wait = _parse_wait_seconds(str(e)) * BACKOFF_MULTIPLIER
                    wait = min(wait, MAX_WAIT_SECONDS)

                    if attempt < MAX_RETRIES_PER_KEY:
                        print(
                            f"  [{key_label}] RPM 限速，等待 {wait:.1f}s 後重試 "
                            f"（第 {attempt}/{MAX_RETRIES_PER_KEY} 次）"
                        )
                        time.sleep(wait)
                    else:
                        print(f"  [{key_label}] RPM 重試 {MAX_RETRIES_PER_KEY} 次仍失敗，換下一個 key")
                        break

                else:
                    # 其他錯誤（網路、認證等）→ 直接換下一個 key
                    logger.warning(f"[{key_label}] 錯誤：{e}")
                    break

    raise RuntimeError(f"所有 API key 都無法使用，最後錯誤：{last_error}")