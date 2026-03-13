import json
import os
from pathlib import Path

DATA_PATH = "data/ingredients.json"
PENDING_PATH = "data/pending_ingredients.json"


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _find_index(ingredients: list, inci_name: str) -> int:
    """找到 inci_name 對應的 index，找不到回傳 -1"""
    for i, item in enumerate(ingredients):
        if item.get("inci_name", "").upper() == inci_name.upper():
            return i
    return -1


def write_to_db(data: dict, verdict: str = "pass") -> dict:
    """
    將通過驗證的成分資料寫入 ingredients.json。
    如果成分已存在則更新，不存在則新增。

    Args:
        data: 成分資料 dict
        verdict: "pass" 或 "unverified"（fail 不應該傳進來）

    回傳：{"action": "updated" | "added", "inci_name": str}
    """
    if verdict == "fail":
        raise ValueError("verdict 為 fail 的資料不應寫入 DB")

    # 清理不需要寫入 DB 的欄位（驗證流程的 metadata）
    clean_data = {k: v for k, v in data.items()
                  if k not in ("confidence", "warning", "is_substance")}

    # 標注 unverified
    if verdict == "unverified":
        clean_data["unverified"] = True
    else:
        clean_data.pop("unverified", None)

    ingredients = _load_json(DATA_PATH)
    inci_name = clean_data.get("inci_name", "")
    idx = _find_index(ingredients, inci_name)

    if idx >= 0:
        ingredients[idx] = clean_data
        action = "updated"
    else:
        ingredients.append(clean_data)
        action = "added"

    _save_json(DATA_PATH, ingredients)
    print(f"[Updater] {action}：{inci_name}")
    return {"action": action, "inci_name": inci_name}


def write_to_pending(data: dict) -> dict:
    """
    將 LLM 即時生成的成分資料寫入 pending_ingredients.json。
    用於線上流程：查不到的成分先存到 pending，等待離線流程驗證。

    回傳：{"action": "added" | "already_exists", "inci_name": str}
    """
    pending = _load_json(PENDING_PATH)
    inci_name = data.get("inci_name", "")
    idx = _find_index(pending, inci_name)

    if idx >= 0:
        # 已在 pending 裡，不重複寫入
        return {"action": "already_exists", "inci_name": inci_name}

    pending.append(data)
    _save_json(PENDING_PATH, pending)
    print(f"[Updater] 寫入 pending：{inci_name}")
    return {"action": "added", "inci_name": inci_name}


def rebuild_index() -> None:
    """
    重建 FAISS 索引。
    在 write_to_db 之後呼叫，確保索引和 DB 同步。
    """
    from rag.retriever import build_index
    print("[Updater] 重建 FAISS 索引...")
    build_index()
    print("[Updater] 索引重建完成")


def process_pending() -> dict:
    """
    離線流程：處理 pending_ingredients.json 裡的所有待驗證資料。
    使用 validator 評估品質，通過則寫入 DB，失敗則丟棄。

    回傳統計資訊：{"passed": int, "unverified": int, "failed": int, "errors": int}
    """
    from rag.validator import validate

    pending = _load_json(PENDING_PATH)
    if not pending:
        print("[Updater] pending 清單為空，無需處理")
        return {"passed": 0, "unverified": 0, "failed": 0, "errors": 0}

    stats = {"passed": 0, "unverified": 0, "failed": 0, "errors": 0}
    remaining = []

    for item in pending:
        inci_name = item.get("inci_name", "unknown")
        try:
            result = validate(item, run_judge=True)
            verdict = result["verdict"]

            if verdict == "pass":
                write_to_db(item, verdict="pass")
                stats["passed"] += 1
                print(f"[Updater] ✅ {inci_name} → 寫入 DB")

            elif verdict == "unverified":
                write_to_db(item, verdict="unverified")
                stats["unverified"] += 1
                print(f"[Updater] ⚠️  {inci_name} → 寫入 DB（標注 unverified）")

            elif verdict in ("fail", "format_error"):
                stats["failed"] += 1
                print(f"[Updater] ❌ {inci_name} → 丟棄（{verdict}）")
                print(f"          問題：{result.get('issues') or result.get('format_errors')}")

        except Exception as e:
            stats["errors"] += 1
            remaining.append(item)
            print(f"[Updater] 處理失敗：{inci_name}，錯誤：{e}")

    # 只保留處理失敗的資料（等下次重試）
    _save_json(PENDING_PATH, remaining)

    print(f"\n[Updater] 處理完成：通過 {stats['passed']}，unverified {stats['unverified']}，"
          f"丟棄 {stats['failed']}，錯誤 {stats['errors']}")
    return stats


if __name__ == "__main__":
    import shutil

    # 測試前先備份 ingredients.json
    if os.path.exists(DATA_PATH):
        shutil.copy(DATA_PATH, DATA_PATH + ".bak")
        print("已備份 ingredients.json")

    # 測試 1：寫入新成分
    print("\n=== 測試 1：write_to_db() 新增成分 ===")
    test_data = {
        "ingredient": "Bakuchiol",
        "inci_name": "Bakuchiol",
        "cas_number": "10309-37-2",
        "functions": ["ANTIOXIDANT", "SKIN CONDITIONING"],
        "benefits": ["減少細紋", "抗氧化", "舒緩肌膚"],
        "risks": ["可能引起輕微過敏"],
        "skin_type": ["所有膚質", "敏感肌"],
        "eu_regulation": "無使用限制",
        "source": ["LLM-generated"],
        "confidence": "medium",    # 這個欄位應被清理掉
        "warning": "AI 生成"       # 這個欄位應被清理掉
    }
    result1 = write_to_db(test_data, verdict="pass")
    print(result1)

    # 測試 2：更新已存在的成分
    print("\n=== 測試 2：write_to_db() 更新已存在成分 ===")
    result2 = write_to_db(test_data, verdict="pass")
    print(result2)

    # 測試 3：寫入 pending
    print("\n=== 測試 3：write_to_pending() ===")
    pending_data = {
        "ingredient": "Resveratrol",
        "inci_name": "Resveratrol",
        "cas_number": "501-36-0",
        "functions": ["ANTIOXIDANT"],
        "benefits": ["抗氧化", "抗老化"],
        "risks": ["高濃度可能刺激"],
        "skin_type": ["熟齡肌"],
        "eu_regulation": "無使用限制",
        "source": ["LLM-generated"],
        "confidence": "medium",
        "warning": "AI 生成"
    }
    result3 = write_to_pending(pending_data)
    print(result3)

    # 測試 4：重複寫入 pending（應回傳 already_exists）
    print("\n=== 測試 4：重複寫入 pending ===")
    result4 = write_to_pending(pending_data)
    print(result4)

    # 還原備份
    if os.path.exists(DATA_PATH + ".bak"):
        shutil.move(DATA_PATH + ".bak", DATA_PATH)
        print("\n已還原 ingredients.json")

    # 清理測試用的 pending 檔案
    if os.path.exists(PENDING_PATH):
        os.remove(PENDING_PATH)
        print("已清理 pending_ingredients.json")