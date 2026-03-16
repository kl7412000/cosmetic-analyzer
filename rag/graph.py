import re
import json
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from rag.retriever import load_retriever
from rag.enricher import enrich_from_name
from rag.updater import write_to_pending


# ─── State ───────────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    input_text: str
    input_image: Optional[str]   # base64
    ingredients: list            # 拆解後的成分名稱列表
    found: list                  # FAISS 找到的成分資料
    not_found: list              # FAISS 找不到的成分名稱
    enriched_data: list          # LLM 生成的成分資料
    results: list                # 最終整合結果
    error: Optional[str]


# ─── Nodes ───────────────────────────────────────────────────────────────────

def ocr_node(state: AnalysisState) -> AnalysisState:
    """
    圖片辨識節點。
    有 input_image 才執行，把辨識結果寫入 input_text。
    """
    if not state.get("input_image"):
        return state
    try:
        from rag.ocr import extract_from_base64
        result = extract_from_base64(state["input_image"])
        print(f"=== OCR 結果 ===")
        print(result)
        ocr_text = result.get("ingredients_text", "")
        if ocr_text:
            return {**state, "input_text": ocr_text}
    except Exception as e:
        return {**state, "error": f"OCR 失敗：{e}"}
    return state


def parser_node(state: AnalysisState) -> AnalysisState:
    """
    成分解析節點。
    把輸入文字拆成單一成分列表。
    支援逗號、換行、中文逗號、分號分隔。
    """
    text = state.get("input_text", "").strip()
    if not text and not state.get("input_image"):
        return {**state, "error": "請輸入成分名稱", "ingredients": []}
    if not text:
        # 有圖片但還沒 OCR 結果，等 OCR 節點處理
        return {**state, "ingredients": []}

    # 分隔符：逗號、換行、中文逗號、分號
    parts = re.split(r"[,\n，;；•、·｜|＆&]+", text)
    ingredients = [p.strip() for p in parts if p.strip()]

    if not ingredients:
        return {**state, "error": "無法解析成分", "ingredients": []}

    return {**state, "ingredients": ingredients}


def query_node(state: AnalysisState) -> AnalysisState:
    """
    FAISS 查詢節點。
    對每個成分做語意搜尋，分成 found 和 not_found 兩組。
    """
    ingredients = state.get("ingredients", [])
    if not ingredients:
        return state

    try:
        retriever = load_retriever(k=1)
    except Exception as e:
        return {**state, "error": f"載入索引失敗：{e}"}

    found = []
    not_found = []

    for name in ingredients:
        try:
            docs = retriever.invoke(name)
            if docs:
                doc = docs[0]
                metadata = doc.metadata

                # 確認搜尋結果和查詢的成分名稱夠接近
                db_name = metadata.get("ingredient", "").upper()
                db_inci = metadata.get("inci_name", "").upper()
                query_upper = name.upper()

                is_match = (
                    query_upper in db_name
                    or query_upper in db_inci
                    or db_name in query_upper
                    or db_inci in query_upper
                )

                if is_match:
                    found.append({**metadata, "_query": name})
                else:
                    not_found.append(name)
            else:
                not_found.append(name)

        except Exception:
            not_found.append(name)

    return {**state, "found": found, "not_found": not_found}


def enrich_node(state: AnalysisState) -> AnalysisState:
    """
    LLM 補充節點。
    對 not_found 的成分用 LLM 即時生成資料，
    並在背景寫入 pending_ingredients.json。
    """
    not_found = state.get("not_found", [])
    if not not_found:
        return {**state, "enriched_data": []}

    enriched_data = []

    for name in not_found:
        try:
            data = enrich_from_name(name, confidence="medium")
            enriched_data.append({**data, "_query": name})

            # 背景寫入 pending（不阻塞主流程，本地流程用，HF Spaces 為唯讀環境會自動跳過)
            try:
                write_to_pending(data)
            except Exception:
                pass  # pending 寫入失敗不影響回傳結果

        except Exception as e:
            # LLM 生成失敗，回傳錯誤佔位
            enriched_data.append({
                "ingredient": name,
                "inci_name": name.upper(),
                "error": f"無法生成資料：{e}",
                "source": ["error"],
                "_query": name,
            })

    return {**state, "enriched_data": enriched_data}


def response_node(state: AnalysisState) -> AnalysisState:
    """
    整合節點。
    合併 found 和 enriched_data，按照原始輸入順序排列結果。
    """
    ingredients = state.get("ingredients", [])
    found = state.get("found", [])
    not_found_names = state.get("not_found", [])
    enriched_data = state.get("enriched_data", [])

    # 建立查詢名稱 → 資料的對照表
    lookup = {}
    for item in found:
        lookup[item.get("_query", "").upper()] = {
            **{k: v for k, v in item.items() if k != "_query"},
            "confidence": "high",
        }
    for item in enriched_data:
        lookup[item.get("_query", "").upper()] = {
            **{k: v for k, v in item.items() if k != "_query"},
            "confidence": item.get("confidence", "medium"),
        }

    # 按照原始輸入順序整合結果
    results = []
    for name in ingredients:
        data = lookup.get(name.upper())
        if data:
            results.append({**data, "_original": name})
        else:
            results.append({
                "ingredient": name,
                "error": "查詢失敗",
                "confidence": "error",
                "_original": name,
            })


    return {**state, "results": results}


# ─── 條件邊：決定是否執行 OCR ─────────────────────────────────────────────────

def should_ocr(state: AnalysisState) -> str:
    if state.get("input_image"):
        return "ocr"
    return "parser"


# ─── 建立 Graph ───────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("parser", parser_node)
    graph.add_node("query", query_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("response", response_node)

    # 入口：根據有無圖片決定走 OCR 還是直接 parser
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


# 建立全域 graph 實例（app.py import 用）
online_graph = build_graph()


def analyze_online(text: str, image_b64: Optional[str] = None) -> list:
    """
    線上流程入口。

    Args:
        text: 成分名稱文字輸入
        image_b64: 圖片 base64（可選）

    回傳：成分分析結果列表
    """
    initial_state: AnalysisState = {
        "input_text": text or "",
        "input_image": image_b64,
        "ingredients": [],
        "found": [],
        "not_found": [],
        "enriched_data": [],
        "results": [],
        "error": None,
    }

    final_state = online_graph.invoke(initial_state)

    if final_state.get("error"):
        return [{"error": final_state["error"]}]

    return final_state.get("results", [])


if __name__ == "__main__":
    print("=== 測試線上流程（文字輸入）===")

    # 測試 1：已知成分（應走 found）
    print("\n--- 測試 1：已知成分 ---")
    results = analyze_online("Niacinamide, Hyaluronic Acid")
    for r in results:
        print(f"{r.get('ingredient', 'unknown')} → confidence: {r.get('confidence')}")
        print(f"  benefits: {r.get('benefits', [])[:2]}")

    # 測試 2：未知成分（應走 enrich）
    print("\n--- 測試 2：未知成分（LLM fallback）---")
    results2 = analyze_online("Bakuchiol")
    for r in results2:
        print(f"{r.get('ingredient', 'unknown')} → confidence: {r.get('confidence')}")
        print(f"  source: {r.get('source')}")
        print(f"  warning: {r.get('warning', '無')}")

    # 測試 3：混合輸入
    print("\n--- 測試 3：混合輸入（已知 + 未知）---")
    results3 = analyze_online("Retinol, Resveratrol")
    for r in results3:
        print(f"{r.get('ingredient', 'unknown')} → confidence: {r.get('confidence')}")