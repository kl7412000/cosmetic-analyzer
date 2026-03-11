from sentence_transformers import SentenceTransformer

# 載入模型（第一次執行會自動下載，約 80MB）
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> list:
    """
    把一段文字轉成向量
    input:  字串
    output: 向量（list of float）
    """
    return model.encode(text).tolist()

def embed_batch(texts: list) -> list:
    """
    把多段文字一次轉成向量（建立索引時用）
    input:  字串 list
    output: 向量 list
    """
    return model.encode(texts).tolist()