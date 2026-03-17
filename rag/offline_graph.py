# rag/offline_graph.py
import json
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from rag.enricher import enrich
from rag.validator import validate
from rag.updater import write_to_db, rebuild_index
from scraper.inci_decoder import scrape as scrape_inci


# ── State 定義 ────────────────────────────────────────────────────────
class OfflineState(TypedDict):
    inci_name: str
    scraped: dict
    enriched: dict
    validation: dict
    verdict: str        # "pass" / "unverified" / "fail" / "format_error"
    error: Optional[str]
    log: list


# ── Nodes ─────────────────────────────────────────────────────────────
def scraper_node(state: OfflineState) -> OfflineState:
    """
    從 INCI Decoder 爬取成分資料。
    爬取失敗時不中止流程，而是回傳空 dict 讓後續節點用 LLM 補充。
    """
    log = state["log"] + []
    try:
        scraped = scrape_inci(state["inci_name"])
        if scraped:
            log.append(f"INCI Decoder: 找到 {state['inci_name']}")
            return {**state, "scraped": scraped, "log": log}
        else:
            log.append(f"INCI Decoder: 找不到 {state['inci_name']}，交由 LLM 補充")
            return {**state, "scraped": {}, "log": log}
    except Exception as e:
        log.append(f"INCI Decoder 爬取失敗：{str(e)}，交由 LLM 補充")
        return {**state, "scraped": {}, "log": log}


def enrich_node(state: OfflineState) -> OfflineState:
    """
    用 LLM 補充缺少的欄位。
    有爬蟲資料時用 enrich()（補充模式），
    沒有爬蟲資料時用 enrich_from_name()（從零生成模式）。
    """
    from rag.enricher import enrich_from_name
    log = state["log"] + []

    try:
        if state["scraped"]:
            enriched = enrich(state["scraped"])
            log.append("Enricher: 補充爬蟲資料完成")
        else:
            enriched = enrich_from_name(state["inci_name"])
            log.append("Enricher: 從零生成完成")

        return {**state, "enriched": enriched, "log": log}

    except Exception as e:
        log.append(f"Enricher 失敗：{str(e)}")
        return {**state, "error": f"Enricher 失敗：{str(e)}", "log": log}


def should_continue_after_enrich(state: OfflineState) -> str:
    """enrich 失敗時直接結束，不進行後續驗證。"""
    return "end" if state.get("error") else "validator"


def validator_node(state: OfflineState) -> OfflineState:
    """
    執行完整驗證（格式驗證 + LLM-as-a-judge）。
    離線流程一律執行 run_judge=True。
    """
    log = state["log"] + []

    try:
        result = validate(state["enriched"], run_judge=True)
        verdict = result["verdict"]
        score = result.get("score", -1)

        log.append(f"Validator: score={score}, verdict={verdict}")
        if result.get("issues"):
            log.append(f"Validator issues: {result['issues']}")

        return {**state, "validation": result, "verdict": verdict, "log": log}

    except Exception as e:
        log.append(f"Validator 失敗：{str(e)}")
        return {**state, "verdict": "format_error", "error": str(e), "log": log}


def indexer_node(state: OfflineState) -> OfflineState:
    """
    根據 verdict 決定是否寫入 DB。
    pass / unverified → 寫入
    fail / format_error → 丟棄
    """
    log = state["log"] + []
    verdict = state["verdict"]

    if verdict in ("pass", "unverified"):
        action = write_to_db(state["enriched"], verdict=verdict)
        log.append(f"Indexer: {action['action']} → {action['ingredient']}")
    else:
        log.append(f"Indexer: 丟棄 {state['inci_name']} (verdict: {verdict})")

    return {**state, "log": log}


# ── Graph 建立 ────────────────────────────────────────────────────────
def build_offline_graph():
    graph = StateGraph(OfflineState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("validator", validator_node)
    graph.add_node("indexer", indexer_node)

    graph.set_entry_point("scraper")
    graph.add_edge("scraper", "enrich")
    graph.add_conditional_edges(
        "enrich",
        should_continue_after_enrich,
        {"validator": "validator", "end": END}
    )
    graph.add_edge("validator", "indexer")
    graph.add_edge("indexer", END)

    return graph.compile()


offline_graph = build_offline_graph()


# ── 對外介面 ──────────────────────────────────────────────────────────
def process_ingredient(inci_name: str) -> dict:
    """
    處理單一成分的完整離線流程。
    回傳處理結果，包含 verdict 和 log。
    """
    initial_state: OfflineState = {
        "inci_name": inci_name,
        "scraped": {},
        "enriched": {},
        "validation": {},
        "verdict": "",
        "error": None,
        "log": [],
    }
    final_state = offline_graph.invoke(initial_state)

    return {
        "inci_name": inci_name,
        "verdict": final_state["verdict"],
        "error": final_state.get("error"),
        "log": final_state["log"],
    }


def process_batch(inci_names: list) -> dict:
    """
    批次處理多個成分，最後統一重建索引。
    比逐一重建索引更有效率。

    回傳處理摘要：{"added": int, "updated": int, "rejected": int, "results": list}
    """
    summary = {"added": 0, "updated": 0, "rejected": 0, "results": []}

    for name in inci_names:
        result = process_ingredient(name)
        summary["results"].append(result)

        verdict = result["verdict"]
        if verdict in ("pass", "unverified"):
            # write_to_db 已在 indexer_node 內執行，這裡只計數
            # 從 log 判斷是 added 還是 updated
            log_str = " ".join(result["log"])
            if "added" in log_str:
                summary["added"] += 1
            elif "updated" in log_str:
                summary["updated"] += 1
        else:
            summary["rejected"] += 1

        # 逐一列印進度
        print(f"  [{verdict}] {name}")
        for line in result["log"]:
            print(f"    {line}")

    # 有任何成功寫入才重建索引
    if summary["added"] + summary["updated"] > 0:
        rebuild_index()

    return summary


# ── 本地測試 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 測試單一成分處理 ===")
    result = process_ingredient("Bakuchiol")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 測試批次處理 ===")
    summary = process_batch(["Allantoin", "Adenosine"])
    print(f"\n處理摘要：added={summary['added']}, updated={summary['updated']}, rejected={summary['rejected']}")