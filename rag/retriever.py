import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag.ingestor import load_ingredients

FAISS_INDEX_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

# 初始化 embedding model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

def build_index() -> None:
    """
    建立 FAISS 索引，儲存至本地
    """
    documents = load_ingredients()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"索引建立完成，共 {len(documents)} 筆資料")

def load_retriever(k: int = 3):
    """
    載入已建立的 FAISS 索引，回傳 LangChain Retriever
    input:  k（回傳幾筆相關資料）
    output: LangChain Retriever 物件
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})