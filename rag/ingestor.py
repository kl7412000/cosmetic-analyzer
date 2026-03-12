import json
from langchain_core.documents import Document

def load_ingredients(path: str = "data/ingredients.json") -> list[Document]:
    """
    讀取 ingredients.json，轉換成 LangChain Document 格式
    input:  JSON 檔案路徑
    output: LangChain Document list
    """
    with open(path, "r", encoding="utf-8") as f:
        ingredients = json.load(f)

    documents = []
    for item in ingredients:
        # 把每筆資料轉成一段文字作為 page_content
        content = f"""
ingredient: {item['ingredient']}
inci_name: {item['inci_name']}
functions: {', '.join(item['functions'])}
benefits: {', '.join(item['benefits'])}
eu_regulation: {item['eu_regulation']}
        """.strip()

        # 把完整的原始資料存在 metadata 裡，方便之後取用
        doc = Document(
            page_content=content,
            metadata=item
        )
        documents.append(doc)

    return documents