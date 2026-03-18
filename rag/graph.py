# rag/graph.py
from dotenv import load_dotenv
load_dotenv()
import json
import time
from typing import Optional
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
    input_image: Optional[str]   # base64 字串，無圖片時為 None
    original_ingredients: list   # 新增：OCR 辨識的原始名稱
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
        ocr_ingredients = result.get("ingredients", [])
        ocr_text = ", ".join(ocr_ingredients)
        combined = ", ".join(filter(None, [state["input_text"], ocr_text]))
        return {
            **state,
            "input_text": combined,
            "original_ingredients": ocr_ingredients  # 保存原始名稱
        }
    except Exception as e:
        return {**state, "error": f"OCR 失敗：{str(e)}"}
    
def normalize_node(state: AnalysisState) -> AnalysisState:
    """
    將 OCR 辨識出的非英文成分名稱統一轉換為 INCI 英文名稱。
    僅在有圖片輸入時執行，純文字輸入不需要此步驟。
    """
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
        print(f"[NORMALIZE] 翻譯結果：{normalized}")

        if normalized:
            normalized_text = ", ".join(normalized)
            return {**state, "input_text": normalized_text}

    except Exception as e:
        print(f"[NORMALIZE] 失敗：{e}")

    return state


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


# 常見同義詞對照（查詢名稱 → DB 中的 ingredient 或 inci_name）
# key 全部小寫，value 是 DB 裡存的名稱
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
    """
    對每個成分查詢 FAISS 索引。
    判斷邏輯（依序）：
      1. 同義詞替換（Fragrance → Parfum 等）
      2. 名稱完全吻合（大小寫不分）→ 直接視為找到，不管 score
      3. score < 1.2 且名稱部分吻合 → 視為找到（考慮同義詞查詢可能距離較遠）
      4. 其他 → not_found，交給 LLM enrich
    """
    found = []
    not_found = []
    vectorstore = get_vectorstore()

    for name in state["ingredients"]:
        # Step 1：同義詞替換
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

            # Step 2：名稱完全吻合
            exact_match = (lookup_lower == matched_lower or lookup_lower == inci_lower)
            # Step 3：score 閾值（同義詞查詢允許更高的 score，因為語義距離可能較遠）
            score_threshold = 1.2 if is_synonym_query else 1.0
            # Step 4：名稱部分吻合
            partial_match = (
                lookup_lower in matched_lower or matched_lower in lookup_lower or
                lookup_lower in inci_lower or inci_lower in lookup_lower
            )

            if exact_match or (score < score_threshold and partial_match):
                metadata = doc.metadata
                source = metadata.get("source", [])
                confidence = "medium" if source == ["LLM-generated"] else "high"
                # _query_name 記錄原始查詢名稱（未替換的），供 response_node 對應用
                found.append({**metadata, "confidence": confidence, "_query_name": name})
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
            time.sleep(1)  # 避免 rate limit

        except Exception as e:
            print(f"[ENRICH] {name} → 失敗：{e}")
            enriched_data.append({
                "ingredient": name,
                "confidence": "error",
                "error": str(e)
            })

    return {**state, "enriched_data": enriched_data}


def response_node(state: AnalysisState) -> AnalysisState:
    # 建立 found_map：ingredient、inci_name、_query_name 都當 key
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
            item = {**found_map[key], "_display_name": original_name}
            results.append(item)
        elif key in enriched_map:
            item = {**enriched_map[key], "_display_name": original_name}
            results.append(item)
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

    print(f"[RESPONSE] ingredients: {state['ingredients']}")
    print(f"[RESPONSE] original_ingredients: {state.get('original_ingredients', [])}")
    print(f"[RESPONSE] found_map keys: {list(found_map.keys())}")
    print(f"[RESPONSE] enriched_map keys: {list(enriched_map.keys())}")

    return {**state, "results": results}


# ── Graph 建立 ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("normalize", normalize_node)
    graph.add_node("parser", parser_node)
    graph.add_node("query", query_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("response", response_node)

    graph.set_conditional_entry_point(
        should_ocr,
        {"ocr": "ocr", "parser": "parser"}
    )
    graph.add_edge("ocr", "normalize")    # ocr → normalize
    graph.add_edge("normalize", "parser") # normalize → parser
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
        "original_ingredients": [],  # 新增
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
    print("=== 測試 score 數值 ===")
    results = analyze_online("グリセリン, AQUA/WATER/EAU, Glycerin, Water, Niacinamide, Bakuchiol")
    for r in results:
        print(f"{r.get('_original') or r.get('ingredient')} → confidence: {r.get('confidence')}")