import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag.ingestor import load_ingredients

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        try:
            logger.info(f"載入 embedding 模型：{MODEL_NAME}")
            _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            logger.info("Embedding 模型載入成功")
        except Exception as e:
            logger.error(f"Embedding 模型載入失敗：{e}")
            raise
    return _embeddings


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.error(f"找不到 FAISS 索引於: {os.path.abspath(FAISS_INDEX_PATH)}")
            raise FileNotFoundError(f"找不到 FAISS 索引，請先執行 python build_index.py")
        try:
            logger.info(f"載入 FAISS 索引從: {FAISS_INDEX_PATH}")
            _vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS 索引載入成功")
        except Exception as e:
            logger.error(f"FAISS 索引載入失敗：{e}")
            raise
    return _vectorstore

def get_vectorstore() -> FAISS:
    return _get_vectorstore()

def build_index() -> None:
    global _vectorstore
    try:
        logger.info("開始構建 FAISS 索引...")
        documents = load_ingredients()
        logger.info(f"已載入 {len(documents)} 筆成分資料")
        vs = FAISS.from_documents(documents, get_embeddings())
        vs.save_local(FAISS_INDEX_PATH)
        _vectorstore = vs  # 重建後同步更新 singleton
        logger.info(f"索引建立完成，共 {len(documents)} 筆資料")
        print(f"✓ 索引建立完成，共 {len(documents)} 筆資料")
    except Exception as e:
        logger.error(f"索引建立失敗：{e}")
        raise

def load_retriever(k: int = 3):
    return _get_vectorstore().as_retriever(search_kwargs={"k": k})