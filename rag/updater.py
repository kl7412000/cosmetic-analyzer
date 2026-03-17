# rag/updater.py
import json
import os
from datetime import datetime

DB_PATH = "data/ingredients.json"
PENDING_PATH = "data/pending_ingredients.json"


def _load_db() -> list:
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


def _save_db(data: list) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_to_db(data: dict, verdict: str = "pass") -> dict:
    """
    將驗證通過的成分資料寫入 ingredients.json。
    依 inci_name 去重：已存在則更新，不存在則新增。

    回傳 {"action": "added" / "updated", "ingredient": str}
    """
    db = _load_db()

    # 寫入前附上 metadata
    data["_verdict"] = verdict
    data["_updated_at"] = datetime.now().strftime("%Y-%m-%d")

    # 以 inci_name 作為唯一鍵做去重
    inci_name = data.get("inci_name", "").strip().lower()
    for i, item in enumerate(db):
        if item.get("inci_name", "").strip().lower() == inci_name:
            db[i] = data
            _save_db(db)
            return {"action": "updated", "ingredient": data.get("ingredient", inci_name)}

    db.append(data)
    _save_db(db)
    return {"action": "added", "ingredient": data.get("ingredient", inci_name)}


def write_to_pending(data: dict) -> None:
    """
    將線上 LLM fallback 生成的資料寫入 pending_ingredients.json，
    等待離線流程定期驗證後再決定是否寫入主知識庫。
    """
    
    pending = []
    if os.path.exists(PENDING_PATH):
        with open(PENDING_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # 只在檔案有內容時才解析
                pending = json.loads(content)
    

    # 避免重複寫入同一個成分
    inci_name = data.get("inci_name", "").strip().lower()
    for item in pending:
        if item.get("inci_name", "").strip().lower() == inci_name:
            return  # 已存在，跳過

    data["_pending_at"] = datetime.now().strftime("%Y-%m-%d")
    pending.append(data)

    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


def rebuild_index() -> None:
    """
    重建 FAISS 索引。
    每次 DB 有變動後呼叫，確保線上查詢使用最新知識庫。
    """
    from rag.retriever import build_index
    build_index()
    print("索引重建完成")


def process_pending() -> dict:
    """
    離線批次處理：讀取 pending_ingredients.json，
    逐一執行 validator，通過則寫入主 DB，最後統一重建索引。

    回傳處理摘要 {"added": int, "updated": int, "rejected": int}
    """
    from rag.validator import validate

    if not os.path.exists(PENDING_PATH):
        print("沒有待處理的 pending 資料")
        return {"added": 0, "updated": 0, "rejected": 0}

    with open(PENDING_PATH, "r", encoding="utf-8") as f:
        pending = json.load(f)

    summary = {"added": 0, "updated": 0, "rejected": 0}
    survived = []  # 未通過的保留在 pending

    for item in pending:
        # pending 資料已經是 LLM 生成的，這裡執行完整的 LLM-as-a-judge
        result = validate(item, run_judge=True)
        verdict = result["verdict"]

        if verdict in ("pass", "unverified"):
            action = write_to_db(item, verdict=verdict)
            summary[action["action"]] += 1
            print(f"  {action['action']} → {action['ingredient']} (verdict: {verdict})")
        else:
            summary["rejected"] += 1
            survived.append(item)
            print(f"  rejected → {item.get('ingredient', '?')} (verdict: {verdict})")

    # 把未通過的保留在 pending，通過的已移入 DB
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(survived, f, ensure_ascii=False, indent=2)

    # 只有真正有寫入時才重建索引
    if summary["added"] + summary["updated"] > 0:
        rebuild_index()

    return summary


# ── 本地測試 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rag.enricher import enrich_from_name

    print("=== 測試 write_to_pending ===")
    test_data = enrich_from_name("Bakuchiol")
    write_to_pending(test_data)
    print(f"已寫入 pending：{test_data['ingredient']}")

    print("\n=== 測試 process_pending ===")
    summary = process_pending()
    print(f"\n處理摘要：{summary}")