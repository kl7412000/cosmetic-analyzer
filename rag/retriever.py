import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag.ingestor import load_ingredients

FAISS_INDEX_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("找不到 FAISS 索引，請先執行 build_index.py")
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return _get_vectorstore()


def build_index() -> None:
    global _vectorstore
    documents = load_ingredients()
    vs = FAISS.from_documents(documents, get_embeddings())
    vs.save_local(FAISS_INDEX_PATH)
    _vectorstore = vs  # 重建後同步更新 singleton
    print(f"索引建立完成，共 {len(documents)} 筆資料")


def load_retriever(k: int = 3):
    return _get_vectorstore().as_retriever(search_kwargs={"k": k})