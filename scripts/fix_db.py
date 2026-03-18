# scripts/fix_db.py
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()  # ← 移到最前面，在其他 rag 模組 import 之前

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.offline_graph import process_ingredient, process_batch
from rag.updater import _load_db, _save_db, rebuild_index, write_to_db



def fix_ingredient(inci_name: str):
    """重新爬取並驗證單一成分，覆蓋 DB 中的舊資料。"""
    print(f"\n=== 修正：{inci_name} ===")
    result = process_ingredient(inci_name)
    print(f"verdict: {result['verdict']}")
    for line in result['log']:
        print(f"  {line}")
    return result


def fix_batch(inci_names: list):
    """批次修正多個成分。"""
    print(f"\n=== 批次修正 {len(inci_names)} 個成分 ===")
    from rag.offline_graph import process_batch
    summary = process_batch(inci_names)
    print(f"\n處理摘要：added={summary['added']}, updated={summary['updated']}, rejected={summary['rejected']}")
    return summary


def remove_ingredient(inci_name: str):
    """從 DB 中刪除指定成分（用 inci_name 或 ingredient 比對）。"""
    db = _load_db()
    original_count = len(db)
    db = [
        item for item in db
        if item.get("inci_name", "").lower() != inci_name.lower()
        and item.get("ingredient", "").lower() != inci_name.lower()
    ]
    if len(db) < original_count:
        _save_db(db)
        rebuild_index()
        print(f"已刪除：{inci_name}，DB 剩餘 {len(db)} 筆")
    else:
        print(f"找不到：{inci_name}")


def list_db():
    """列出 DB 中所有成分及其來源。"""
    db = _load_db()
    print(f"\n=== DB 目前共 {len(db)} 筆成分 ===")
    for item in db:
        source = item.get("source", [])
        verdict = item.get("_verdict", "—")
        confidence = "⚠️  LLM" if source == ["LLM-generated"] else "✅ 官方"
        print(f"  {confidence}  {item.get('ingredient', '?'):30s}  verdict:{verdict}")


def fix_llm_generated():
    """找出所有 source 為 LLM-generated 的成分，重新走離線流程修正。"""
    db = _load_db()
    llm_items = [
        item.get("inci_name") or item.get("ingredient")
        for item in db
        if item.get("source") == ["LLM-generated"]
    ]
    if not llm_items:
        print("DB 中沒有 LLM-generated 的成分")
        return
    print(f"找到 {len(llm_items)} 個 LLM-generated 成分：{llm_items}")
    fix_batch(llm_items)
    
def add_manual(data: dict):
    """直接寫入一筆成分資料，不經過爬蟲流程。"""
    from rag.validator import validate
    result = validate(data, run_judge=True)
    print(f"verdict: {result['verdict']}, score: {result['score']}")
    if result['verdict'] in ('pass', 'unverified'):
        action = write_to_db(data, verdict=result['verdict'])
        rebuild_index()
        print(f"{action['action']} → {action['ingredient']}")
    else:
        print(f"驗證失敗：{result['issues']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DB 維護工具")
    parser.add_argument("action", choices=["fix", "fix-batch", "remove", "list", "fix-llm", "add-manual"],
                    help="執行動作")
    parser.add_argument("--names", nargs="+", help="成分 INCI 名稱（可多個）")
    args = parser.parse_args()

    if args.action == "fix":
        if not args.names:
            print("請提供 --names 參數")
        else:
            for name in args.names:
                fix_ingredient(name)

    elif args.action == "fix-batch":
        if not args.names:
            print("請提供 --names 參數")
        else:
            fix_batch(args.names)

    elif args.action == "remove":
        if not args.names:
            print("請提供 --names 參數")
        else:
            for name in args.names:
                remove_ingredient(name)

    elif args.action == "list":
        list_db()

    elif args.action == "fix-llm":
        fix_llm_generated()
    elif args.action == "add-manual":
        water = {
            "ingredient": "Water",
            "inci_name": "Aqua",
            "cas_number": "7732-18-5",
            "functions": ["Solvent", "Diluent", "Humectant"],
            "benefits": [
                "Dissolves and delivers other ingredients",
                "Adjusts product consistency",
                "Hydrates skin surface"
            ],
            "risks": [],
            "skin_type": ["All skin types"],
            "eu_regulation": "No restrictions",
            "source": ["CosIng", "Paula's Choice"]
        }
        add_manual(water)