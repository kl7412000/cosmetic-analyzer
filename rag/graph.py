# rag/graph.py
import json
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from rag.chain import parse_ingredients
from rag.retriever import load_retriever
from rag.enricher import enrich_from_name
from rag.updater import write_to_pending
from rag.validator import validate_format
from rag.ocr import extract_from_base64

# ── State 定義 ────────────────────────────────────────────────────────
class AnalysisState(TypedDict):
    input_text: str
    input_image: Optional[str]   # base64 字串，無圖片時為 None
    ingredients: list        # parser_node 拆出的成分列表
    found: list              # query_node 找到的成分（來自 DB）
    not_found: list          # query_node 找不到的成分名稱
    enriched_data: list      # enrich_node 生成的 LLM fallback 資料
    results: list            # response_node 整合後的最終結果
    error: Optional[str]



# ── Nodes ─────────────────────────────────────────────────────────────

def ocr_node(state: AnalysisState) -> AnalysisState:
    """
    有圖片時執行 OCR，將辨識結果合併進 input_text。
    若原本 input_text 也有內容，兩者合併後一起進入 parser_node。
    """
    try:
        result = extract_from_base64(state["input_image"])
        ocr_text = ", ".join(result.get("ingredients", []))
        # 合併文字輸入與 OCR 結果
        combined = ", ".join(filter(None, [state["input_text"], ocr_text]))
        return {**state, "input_text": combined}
    except Exception as e:
        return {**state, "error": f"OCR 失敗：{str(e)}"}


def should_ocr(state: AnalysisState) -> str:
    """條件入口：有圖片走 ocr，否則直接走 parser。"""
    return "ocr" if state.get("input_image") else "parser"


def parser_node(state: AnalysisState) -> AnalysisState:
    """
    把輸入文字拆成單一成分列表。
    使用 chain.py 已有的 parse_ingredients()，保持邏輯一致。
    """
    ingredients = parse_ingredients(state["input_text"])
    return {**state, "ingredients": ingredients}


def query_node(state: AnalysisState) -> AnalysisState:
    """
    對每個成分查詢 FAISS 索引。
    相似度判斷：取 top-1 結果，若成分名稱字串相符則視為找到。
    找到的放入 found，找不到的放入 not_found。
    """
    retriever = load_retriever(k=1)
    found = []
    not_found = []

    for name in state["ingredients"]:
        docs = retriever.invoke(name)
        if docs:
            metadata = docs[0].metadata
            db_names = [
                metadata.get("ingredient", "").lower(),
                metadata.get("inci_name", "").lower(),
            ]
            if name.lower() in db_names:
                # 根據來源決定 confidence，而非一律 high
                source = metadata.get("source", [])
                if source == ["LLM-generated"]:
                    confidence = "medium"
                else:
                    confidence = "high"
                found.append({**metadata, "confidence": confidence})
            else:
                not_found.append(name)
        else:
            not_found.append(name)

    return {**state, "found": found, "not_found": not_found}


def enrich_node(state: AnalysisState) -> AnalysisState:
    """
    對 not_found 的成分呼叫 LLM fallback。
    生成資料通過格式驗證後：
    - 回傳給用戶（confidence: medium）
    - 背景寫入 pending_ingredients.json
    """
    enriched_data = []

    for name in state["not_found"]:
        try:
            data = enrich_from_name(name)

            passed, errors = validate_format(data)
            if not passed:
                enriched_data.append({
                    "ingredient": name,
                    "confidence": "error",
                    "error": f"格式驗證失敗：{errors}"
                })
                continue

            # 保留原始查詢名稱，供 response_node 比對用
            data["_original"] = name

            try:
                write_to_pending(data)
            except Exception:
                pass

            enriched_data.append(data)

        except Exception as e:
            enriched_data.append({
                "ingredient": name,
                "confidence": "error",
                "error": str(e)
            })

    return {**state, "enriched_data": enriched_data}


def response_node(state: AnalysisState) -> AnalysisState:
    found_map = {
        item.get("ingredient", "").lower(): item
        for item in state["found"]
    }
    found_map.update({
        item.get("inci_name", "").lower(): item
        for item in state["found"]
    })

    enriched_map = {}
    for item in state["enriched_data"]:
        # 用 _original（原始查詢名稱）作為主要 key
        if item.get("_original"):
            enriched_map[item["_original"].lower()] = item
        # 同時用 ingredient 作為備用 key
        if item.get("ingredient"):
            enriched_map[item["ingredient"].lower()] = item

    results = []
    for name in state["ingredients"]:
        key = name.lower()
        if key in found_map:
            results.append(found_map[key])
        elif key in enriched_map:
            results.append(enriched_map[key])
        else:
            results.append({
                "ingredient": name,
                "confidence": "error",
                "error": "查詢失敗，請稍後再試"
            })

    return {**state, "results": results}


# ── Graph 建立 ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("parser", parser_node)
    graph.add_node("query", query_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("response", response_node)

    graph.set_conditional_entry_point(
        should_ocr,
        {"ocr": "ocr", "parser": "parser"}
    )
    graph.add_edge("ocr", "parser")
    graph.add_edge("parser", "query")
    graph.add_edge("query", "enrich")
    graph.add_edge("enrich", "response")
    graph.add_edge("response", END)

    return graph.compile()

online_graph = build_graph() 

def analyze_online(text: str, image_b64: str = None) -> list:
    initial_state: AnalysisState = {
        "input_text": text,
        "input_image": image_b64,
        "ingredients": [],
        "found": [],
        "not_found": [],
        "enriched_data": [],
        "results": [],
        "error": None,
    }
    final_state = online_graph.invoke(initial_state)
    return final_state["results"]


# ── 本地測試 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 測試 1：DB 裡有的成分（應為 confidence: high）===")
    results = analyze_online("Niacinamide")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== 測試 2：DB 裡沒有的成分（應為 confidence: medium）===")
    results = analyze_online("Bakuchiol")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== 測試 3：混合查詢 ===")
    results = analyze_online("Niacinamide, Bakuchiol, Retinol")
    for r in results:
        print(f"{r.get('ingredient')} → confidence: {r.get('confidence')}")