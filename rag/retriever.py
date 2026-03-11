import json
import faiss
import numpy as np
from rag.embedder import embed_text, embed_batch

FAISS_INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/metadata.json"

def build_index(ingredients: list) -> None:
    """
    把所有成分資料轉成向量，建立 FAISS 索引
    input:  ingredients.json 讀進來的 list
    output: 儲存 index.faiss 和 metadata.json 到 faiss_index/
    """
    # 把每筆資料轉成一段文字，方便 embedding
    texts = []
    for item in ingredients:
        text = f"""
        ingredient: {item['ingredient']}
        inci_name: {item['inci_name']}
        functions: {', '.join(item['functions'])}
        benefits: {', '.join(item['benefits'])}
        eu_regulation: {item['eu_regulation']}
        """.strip()
        texts.append(text)

    # 轉成向量
    vectors = np.array(embed_batch(texts)).astype("float32")

    # 建立 FAISS 索引
    dimension = vectors.shape[1]  # 384（all-MiniLM-L6-v2 的向量維度）
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # 儲存索引和原始資料
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(ingredients, f, ensure_ascii=False, indent=2)

    print(f"索引建立完成，共 {len(ingredients)} 筆資料")


def search(query: str, k: int = 3) -> list:
    """
    用問題搜尋最相關的 k 筆資料
    input:  query（用戶輸入的成分名稱）、k（回傳幾筆）
    output: 最相關的 k 筆成分資料
    """
    # 載入索引和 metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 把問題轉成向量
    query_vector = np.array([embed_text(query)]).astype("float32")

    # 搜尋最相近的 k 筆
    _, indices = index.search(query_vector, k)

    # 回傳對應的原始資料
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    return results