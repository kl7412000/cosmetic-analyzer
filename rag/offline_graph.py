import json
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from rag.enricher import enrich
from rag.validator import validate
from rag.updater import write_to_db, rebuild_index


# ─── State ───────────────────────────────────────────────────────────────────

class OfflineState(TypedDict):
    inci_name: str
    scraped: dict
    enriched: dict
    validation: dict
    verdict: str
    error: Optional[str]
    log: list


# ─── Nodes ───────────────────────────────────────────────────────────────────

def _normalize_source(source) -> list:
    if isinstance(source, list):
        return source
    if isinstance(source, str):
        return [source]
    return []


def scraper_node(state: OfflineState) -> OfflineState:
    inci_name = state["inci_name"]
    log = state.get("log", [])
    scraped = {}

    # INCI Decoder
    try:
        from scraper.inci_decoder import scrape as inci_scrape
        inci_result = inci_scrape(inci_name)
        if inci_result:
            scraped.update(inci_result)
            log.append(f"INCI Decoder: 找到 {inci_name}")
        else:
            log.append(f"INCI Decoder: 找不到 {inci_name}")
    except Exception as e:
        log.append(f"INCI Decoder: 失敗 - {e}")

    # CosIng
    try:
        from scraper.cosing import scrape as cosing_scrape
        cosing_result = cosing_scrape(inci_name)
        if cosing_result:
            for key, val in cosing_result.items():
                if key == "functions":
                    existing = [f.upper() for f in scraped.get("functions", [])]
                    for f in val:
                        if f.upper() not in existing:
                            scraped.setdefault("functions", []).append(f)
                elif key == "source":
                    existing_sources = _normalize_source(scraped.get("source", []))
                    for s in _normalize_source(val):
                        if s not in existing_sources:
                            existing_sources.append(s)
                    scraped["source"] = existing_sources
                else:
                    scraped[key] = val
            log.append(f"CosIng: 找到 {inci_name}")
        else:
            log.append(f"CosIng: 找不到 {inci_name}")
    except Exception as e:
        log.append(f"CosIng: 失敗 - {e}")

    # 確保 source 是 list
    scraped["source"] = _normalize_source(scraped.get("source", []))

    if not scraped or not scraped.get("source"):
        return {**state, "error": f"所有爬蟲都找不到：{inci_name}", "log": log}

    return {**state, "scraped": scraped, "log": log}


def enrich_node(state: OfflineState) -> OfflineState:
    if state.get("error"):
        return state

    log = state.get("log", [])
    scraped = state.get("scraped", {})

    try:
        enriched = enrich(scraped)
        log.append("Enricher: 補充完成")
        return {**state, "enriched": enriched, "log": log}
    except Exception as e:
        log.append(f"Enricher: 失敗 - {e}")
        return {**state, "error": f"LLM 補充失敗：{e}", "log": log}


def validator_node(state: OfflineState) -> OfflineState:
    if state.get("error"):
        return state

    log = state.get("log", [])
    enriched = state.get("enriched", {})

    try:
        validation = validate(enriched, run_judge=True)
        verdict = validation["verdict"]
        score = validation.get("score", -1)
        log.append(f"Validator: score={score}, verdict={verdict}")
        if validation.get("issues"):
            log.append(f"Validator issues: {validation['issues']}")
        return {**state, "validation": validation, "verdict": verdict, "log": log}
    except Exception as e:
        log.append(f"Validator: 失敗 - {e}")
        return {**state, "error": f"驗證失敗：{e}", "log": log}


def indexer_node(state: OfflineState) -> OfflineState:
    if state.get("error"):
        return state

    log = state.get("log", [])
    verdict = state.get("verdict", "fail")
    enriched = state.get("enriched", {})
    inci_name = state["inci_name"]

    if verdict in ("pass", "unverified"):
        try:
            result = write_to_db(enriched, verdict=verdict)
            log.append(f"Indexer: {result['action']} → {inci_name}")
        except Exception as e:
            log.append(f"Indexer: 寫入失敗 - {e}")
            return {**state, "error": f"寫入 DB 失敗：{e}", "log": log}
    else:
        log.append(f"Indexer: 丟棄 {inci_name}（verdict={verdict}）")

    return {**state, "log": log}


# ─── 條件邊 ───────────────────────────────────────────────────────────────────

def check_error(state: OfflineState) -> str:
    return "end" if state.get("error") else "continue"


# ─── 建立 Graph ───────────────────────────────────────────────────────────────

def build_offline_graph():
    graph = StateGraph(OfflineState)
    graph.add_node("scraper", scraper_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("validator", validator_node)
    graph.add_node("indexer", indexer_node)
    graph.set_entry_point("scraper")
    graph.add_conditional_edges("scraper", check_error, {"continue": "enrich", "end": END})
    graph.add_conditional_edges("enrich", check_error, {"continue": "validator", "end": END})
    graph.add_conditional_edges("validator", check_error, {"continue": "indexer", "end": END})
    graph.add_edge("indexer", END)
    return graph.compile()


offline_graph = build_offline_graph()


def process_ingredient(inci_name: str, rebuild: bool = False) -> dict:
    """離線流程入口：處理單一成分"""
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

    if rebuild and final_state.get("verdict") in ("pass", "unverified"):
        rebuild_index()

    return {
        "inci_name": inci_name,
        "verdict": final_state.get("verdict", "error"),
        "error": final_state.get("error"),
        "log": final_state.get("log", []),
    }


def process_batch(inci_names: list) -> dict:
    """離線流程入口：批次處理多個成分，最後統一重建索引"""
    stats = {"pass": 0, "unverified": 0, "fail": 0, "error": 0}

    for name in inci_names:
        print(f"\n{'='*50}")
        print(f"處理：{name}")
        result = process_ingredient(name, rebuild=False)
        verdict = result.get("verdict", "error")
        stats[verdict] = stats.get(verdict, 0) + 1
        for log_line in result.get("log", []):
            print(f"  {log_line}")
        if result.get("error"):
            print(f"  ❌ 錯誤：{result['error']}")

    if stats["pass"] + stats["unverified"] > 0:
        print("\n重建 FAISS 索引...")
        rebuild_index()

    print(f"\n{'='*50}")
    print(f"批次完成：pass={stats['pass']}, unverified={stats['unverified']}, "
          f"fail={stats['fail']}, error={stats.get('error', 0)}")
    return stats


if __name__ == "__main__":
    print("=== 測試離線流程（單一成分）===")
    result = process_ingredient("Bakuchiol", rebuild=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))