---
title: Cosmetic Ingredient Analyzer
emoji: 🧴
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.23.3"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🧴 Cosmetic Ingredient Analyzer

基於 **LangGraph Multi-Agent + RAG** 的化妝品成分分析系統。

## Architecture

```
Supervisor Agent（LangGraph）
├── OCR Agent          圖片轉成分文字        Groq Vision
├── Normalize Agent    非英文名稱 → INCI     Groq LLM
├── Parser Agent       拆解成分列表          純 Python
├── Query Agent        FAISS 向量資料庫查詢  CosIng / INCI Decoder
├── Enrich Agent       未知成分 LLM 補全     Groq LLM
└── Response Agent     整合最終輸出          純 Python
```

Supervisor 根據輸入類型動態決定流程：
- 純文字輸入 → Parser → Query → Enrich → Response
- 圖片輸入 → OCR → Normalize → Parser → Query → Enrich → Response

## Features

- **Multi-Agent Pipeline**：LangGraph Supervisor 模式，各 Agent 分工明確
- **即時狀態顯示**：UI 可見每個 Agent 的執行進度
- **成分分析**：FAISS 向量資料庫相似度匹配，資料來源 CosIng / INCI Decoder
- **AI 增強**：資料庫查無的成分由 Groq LLM 自動補全
- **OCR 辨識**：上傳化妝品標籤圖片自動辨識成分
- **多語言支援**：日文、韓文、中文成分名稱自動轉換為 INCI 英文名稱
- **產品比較**：支援兩個產品的成分差異比較

## Tech Stack

| 層級 | 技術 |
|------|------|
| Agent 框架 | LangGraph |
| LLM | Groq（llama-3.3-70b / llama-4-scout） |
| 向量資料庫 | FAISS + HuggingFace Embeddings |
| UI | Gradio |
| OCR | Groq Vision API |

## Setup（本地開發）

1. Clone 倉庫：
   ```bash
   git clone https://github.com/yourusername/cosmetic-analyzer.git
   cd cosmetic-analyzer
   ```

2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

3. 設置環境變數（建立 `.env` 檔案）：
   ```bash
   GROQ_API_KEY_1=your_api_key_here
   GROQ_API_KEY_2=your_api_key_here
   GROQ_API_KEY_3=your_api_key_here
   ```

4. 構建 FAISS 索引：
   ```bash
   python build_index.py
   ```

5. 運行應用：
   ```bash
   python app.py
   ```

## Hugging Face 部署

Secrets 設定（Space Settings → Variables and secrets）：
```
GROQ_API_KEY_1=your_api_key_here
GROQ_API_KEY_2=your_api_key_here
GROQ_API_KEY_3=your_api_key_here
```

## Data Sources

- [CosIng](https://ec.europa.eu/growth/tools-databases/cosing/) - 歐盟化妝品成分資料庫
- [INCI Decoder](https://incidecoder.com/) - 成分解碼資料庫

## License

MIT License