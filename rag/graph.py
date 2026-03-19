# rag/graph.py
from dotenv import load_dotenv
load_dotenv()
import json
import time
from typing import Optional, Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS as FAISSStore

from rag.chain import parse_ingredients
from rag.retriever import get_vectorstore
from rag.enricher import enrich_from_name
from rag.updater import write_to_pending
from rag.validator import validate_format
from rag.ocr import extract_from_base64


# ── State 定義 ────────────────────────────────────────────────────────
class AnalysisState(TypedDict):
    input_text: str
    input_image: Optional[str]      # base64 字串，無圖片時為 None
    original_ingredients: list      # OCR 辨識的原始名稱
    ingredients: list               # parser_node 拆出的成分列表
    found: list                     # query_node 找到的成分（來自 DB）
    not_found: list                 # query_node 找不到的成分名稱
    enriched_data: list             # enrich_node 生成的 LLM fallback 資料
    results: list                   # response_node 整合後的最終結果
    error: Optional[str]

    # ── Supervisor 新增欄位 ──────────────────────────────────────────
    next_node: Optional[str]        # Supervisor 決定下一步要去哪個 node
    agent_status: dict              # 各 Agent 的執行狀態，供 UI 顯示用
    # agent_status 格式：
    # {
    #   "ocr":       "pending" | "running" | "done" | "skip",
    #   "normalize": "pending" | "running" | "done" | "skip",
    #   "parser":    "pending" | "running" | "done" | "skip",
    #   "query":     "pending" | "running" | "done" | "skip",
    #   "enrich":    "pending" | "running" | "done" | "skip",
    #   "response":  "pending" | "running" | "done" | "skip",
    # }


# ── Supervisor Node ───────────────────────────────────────────────────
# Supervisor 是整個 pipeline 的指揮中心。
# 每個 node 執行完畢後都會回到 Supervisor，
# 由 Supervisor 決定下一步要去哪個 node。
# 這是 LangGraph Supervisor 模式的核心設計。

PIPELINE_WITH_IMAGE    = ["ocr", "normalize", "parser", "query", "enrich", "response"]
PIPELINE_WITHOUT_IMAGE = ["parser", "query", "enrich", "response"]

def supervisor_node(state: AnalysisState) -> AnalysisState:
    """
    Supervisor：根據目前狀態決定下一個要執行的 node。

    決策邏輯：
    1. 有圖片 → 走 OCR pipeline
    2. 無圖片 → 跳過 OCR/Normalize，直接走 Parser pipeline
    3. 若發生 error → 直接跳到 response，避免繼續執行
    4. 每個 node 執行完畢後更新 agent_status → done
    """
    status = dict(state.get("agent_status", {}))
    has_image = bool(state.get("input_image"))
    pipeline = PIPELINE_WITH_IMAGE if has_image else PIPELINE_WITHOUT_IMAGE

    # 若有錯誤，直接跳到 response
    if state.get("error"):
        status["response"] = "running"
        return {**state, "next_node": "response", "agent_status": status}

    # 找出目前 pipeline 中第一個尚未執行（pending）的 node
    for node in pipeline:
        if status.get(node) == "pending":
            # 跳過不在此 pipeline 的 node（例如無圖時跳過 ocr/normalize）
            if node not in pipeline:
                status[node] = "skip"
                continue
            status[node] = "running"
            return {**state, "next_node": node, "agent_status": status}

    # 所有 node 都完成了 → 結束
    return {**state, "next_node": END, "agent_status": status}


def route_from_supervisor(state: AnalysisState) -> str:
    """LangGraph conditional edge：從 Supervisor 決定路由。"""
    return state.get("next_node", END)


# ── 初始化 agent_status ───────────────────────────────────────────────
def init_status(has_image: bool) -> dict:
    """根據是否有圖片，初始化各 Agent 的狀態。"""
    if has_image:
        return {
            "ocr":       "pending",
            "normalize": "pending",
            "parser":    "pending",
            "query":     "pending",
            "enrich":    "pending",
            "response":  "pending",
        }
    else:
        return {
            "ocr":       "skip",
            "normalize": "skip",
            "parser":    "pending",
            "query":     "pending",
            "enrich":    "pending",
            "response":  "pending",
        }


# ── 包裝 Node：執行完畢後更新 status → done，再回 Supervisor ──────────
def wrap_node(name: str, fn: Callable) -> Callable:
    """
    高階函式：包裝原本的 node function。
    執行完畢後自動將 agent_status[name] 改為 "done"。
    讓每個 node 不需要自己管理狀態，保持單一職責。
    """
    def wrapped(state: AnalysisState) -> AnalysisState:
        result = fn(state)
        status = dict(result.get("agent_status", state.get("agent_status", {})))
        status[name] = "done"
        return {**result, "agent_status": status}
    return wrapped


# ── Nodes（邏輯與原本完全相同，不做任何修改）─────────────────────────

def ocr_node(state: AnalysisState) -> AnalysisState:
    """有圖片時執行 OCR，將辨識結果合併進 input_text。"""
    try:
        result = extract_from_base64(state["input_image"])
        ocr_ingredients = result.get("ingredients", [])
        ocr_text = ", ".join(ocr_ingredients)
        combined = ", ".join(filter(None, [state["input_text"], ocr_text]))
        return {
            **state,
            "input_text": combined,
            "original_ingredients": ocr_ingredients
        }
    except Exception as e:
        return {**state, "error": f"OCR 失敗：{str(e)}"}


def normalize_node(state: AnalysisState) -> AnalysisState:
    """將 OCR 辨識出的非英文成分名稱統一轉換為 INCI 英文名稱。"""
    raw_text = state.get("input_text", "").strip()
    if not raw_text:
        return state

    print(f"[NORMALIZE] 輸入：{raw_text[:100]}")

    from rag.groq_client import call_groq

    prompt = f"""
    以下是從化妝品成分標籤辨識出的成分列表，可能包含日文、韓文、中文或其他語言：

    {raw_text}

    請將每個成分名稱翻譯並對應到正確的 INCI 英文名稱。
    請嚴格按照以下 JSON 格式輸出，不要加任何額外說明：
    {{
    "normalized": ["INCI名稱1", "INCI名稱2", "INCI名稱3"]
    }}

    注意：
    - 若成分名稱已經是英文 INCI 名稱，直接保留原文
    - 若無法對應到 INCI 名稱，保留原文
    - 輸出順序必須與輸入順序一致
    """

    try:
        result = call_groq(prompt)
        normalized = result.get("normalized", [])
        if normalized:
            normalized_text = ", ".join(normalized)
            return {**state, "input_text": normalized_text}
    except Exception as e:
        print(f"[NORMALIZE] 失敗：{e}")

    return state


def parser_node(state: AnalysisState) -> AnalysisState:
    """把輸入文字拆成單一成分列表。"""
    ingredients = parse_ingredients(state["input_text"])
    return {**state, "ingredients": ingredients}


SYNONYMS = {
    "fragrance": "parfum",
    "aqua": "water",
    "eau": "water",
    "aqua/water/eau": "water",
    "vitamin c": "ascorbic acid",
    "vitamin e": "tocopherol",
    "vit. c": "ascorbic acid",
    "vit. e": "tocopherol",
}


def query_node(state: AnalysisState) -> AnalysisState:
    """對每個成分查詢 FAISS 索引。"""
    found = []
    not_found = []
    vectorstore = get_vectorstore()

    for name in state["ingredients"]:
        lookup_name = SYNONYMS.get(name.lower(), name)
        is_synonym_query = (lookup_name.lower() != name.lower())

        results = vectorstore.similarity_search_with_score(lookup_name, k=1)

        if results:
            doc, score = results[0]
            matched_ingredient = doc.metadata.get("ingredient", "")
            matched_inci = doc.metadata.get("inci_name", "")
            lookup_lower = lookup_name.lower()
            matched_lower = matched_ingredient.lower()
            inci_lower = matched_inci.lower()

            print(f"[DEBUG] {name} → lookup: {lookup_name}, score: {score}, matched: {matched_ingredient}")

            exact_match = (lookup_lower == matched_lower or lookup_lower == inci_lower)
            score_threshold = 1.2 if is_synonym_query else 1.0
            partial_match = (
                lookup_lower in matched_lower or matched_lower in lookup_lower or
                lookup_lower in inci_lower or inci_lower in lookup_lower
            )

            if exact_match or (score < score_threshold and partial_match):
                metadata = doc.metadata
                source = metadata.get("source", [])
                confidence = "medium" if source == ["LLM-generated"] else "high"
                found.append({**metadata, "confidence": confidence, "_query_name": name})
            else:
                not_found.append(name)
        else:
            not_found.append(name)

    return {**state, "found": found, "not_found": not_found}


def enrich_node(state: AnalysisState) -> AnalysisState:
    """對 not_found 的成分呼叫 LLM fallback。"""
    enriched_data = []
    print(f"[ENRICH] not_found: {state['not_found']}")

    for name in state["not_found"]:
        try:
            data = enrich_from_name(name)
            passed, errors = validate_format(data)
            print(f"[ENRICH] {name} → ingredient: {data.get('ingredient')}, passed: {passed}")

            if not passed:
                enriched_data.append({
                    "ingredient": name,
                    "confidence": "error",
                    "error": f"格式驗證失敗：{errors}"
                })
                continue

            data["_original"] = name
            try:
                write_to_pending(data)
            except Exception:
                pass

            enriched_data.append(data)
            time.sleep(1)

        except Exception as e:
            print(f"[ENRICH] {name} → 失敗：{e}")
            enriched_data.append({
                "ingredient": name,
                "confidence": "error",
                "error": str(e)
            })

    return {**state, "enriched_data": enriched_data}


def response_node(state: AnalysisState) -> AnalysisState:
    """整合 found + enriched_data 成最終結果。"""
    found_map = {}
    for item in state["found"]:
        if item.get("ingredient"):
            found_map[item["ingredient"].lower()] = item
        if item.get("inci_name"):
            found_map[item["inci_name"].lower()] = item
        if item.get("_query_name"):
            found_map[item["_query_name"].lower()] = item

    enriched_map = {}
    for item in state["enriched_data"]:
        if item.get("_original"):
            enriched_map[item["_original"].lower()] = item
        if item.get("ingredient"):
            enriched_map[item["ingredient"].lower()] = item

    original_ingredients = state.get("original_ingredients", [])

    results = []
    for i, name in enumerate(state["ingredients"]):
        key = name.lower()
        original_name = original_ingredients[i] if i < len(original_ingredients) else name

        if key in found_map:
            results.append({**found_map[key], "_display_name": original_name})
        elif key in enriched_map:
            results.append({**enriched_map[key], "_display_name": original_name})
        else:
            matched = next(
                (v for v in state["enriched_data"] if v.get("_original", "").lower() == key),
                None
            )
            if matched:
                results.append({**matched, "_display_name": original_name})
            else:
                results.append({
                    "ingredient": name,
                    "_display_name": original_name,
                    "confidence": "error",
                    "error": "查詢失敗，請稍後再試"
                })

    return {**state, "results": results}


# ── Graph 建立 ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AnalysisState)

    # 註冊 Supervisor
    graph.add_node("supervisor", supervisor_node)

    # 註冊各 Agent Node（用 wrap_node 包裝，執行完自動更新 status）
    graph.add_node("ocr",       wrap_node("ocr",       ocr_node))
    graph.add_node("normalize", wrap_node("normalize", normalize_node))
    graph.add_node("parser",    wrap_node("parser",    parser_node))
    graph.add_node("query",     wrap_node("query",     query_node))
    graph.add_node("enrich",    wrap_node("enrich",    enrich_node))
    graph.add_node("response",  wrap_node("response",  response_node))

    # 起點是 Supervisor
    graph.set_entry_point("supervisor")

    # Supervisor → 條件路由到各 node
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "ocr":       "ocr",
            "normalize": "normalize",
            "parser":    "parser",
            "query":     "query",
            "enrich":    "enrich",
            "response":  "response",
            END:         END,
        }
    )

    # 每個 node 執行完畢後都回到 Supervisor
    for node in ["ocr", "normalize", "parser", "query", "enrich", "response"]:
        graph.add_edge(node, "supervisor")

    return graph.compile()


online_graph = build_graph()


def analyze_online(
    text: str,
    image_b64: str = None,
    status_callback: Callable[[dict], None] = None,
) -> list:
    """
    執行完整分析 pipeline。

    Args:
        text:            用戶輸入的成分文字
        image_b64:       圖片 base64（可選）
        status_callback: 每次 Supervisor 更新 agent_status 時的回呼函式
                         供 app.py 的 UI 即時更新 Agent 狀態列使用
    """
    has_image = bool(image_b64)

    initial_state: AnalysisState = {
        "input_text":           text,
        "input_image":          image_b64,
        "original_ingredients": [],
        "ingredients":          [],
        "found":                [],
        "not_found":            [],
        "enriched_data":        [],
        "results":              [],
        "error":                None,
        "next_node":            None,
        "agent_status":         init_status(has_image),
    }

    # 若有 callback，使用 stream 模式讓 UI 能即時更新
    if status_callback:
        final_state = initial_state
        for chunk in online_graph.stream(initial_state):
            for node_name, node_state in chunk.items():
                if node_state.get("agent_status"):
                    status_callback(node_state["agent_status"])
                final_state = node_state
        return final_state.get("results", [])

    # 無 callback，直接 invoke（原本行為，向下相容）
    final_state = online_graph.invoke(initial_state)
    return final_state["results"]


# ── 本地測試 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    def print_status(status: dict):
        icons = {"pending": "⬜", "running": "⏳", "done": "✅", "skip": "—"}
        parts = [f"{icons.get(v, '?')} {k}" for k, v in status.items()]
        print(" → ".join(parts))

    print("=== 測試 Supervisor Pipeline ===")
    results = analyze_online(
        "Glycerin, Niacinamide, Bakuchiol",
        status_callback=print_status
    )
    for r in results:
        print(f"{r.get('_display_name') or r.get('ingredient')} → confidence: {r.get('confidence')}")