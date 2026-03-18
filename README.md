---
title: Cosmetic Ingredient Analyzer
emoji: 🧴
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Cosmetic Ingredient Analyzer

基於 RAG 的化妝品成分分析系統，輸入成分名稱獲取功效、法規資訊與來源說明。

## Features
- 成分分析：使用 FAISS 向量數據庫進行相似度匹配
- 知識增強：通過 Groq LLM 補充未找到的成分資訊
- OCR 識別：支持上傳化妝品標籤圖片進行成分識別
- 多語言支持：支持英文、日文、韓文、中文等成分名稱

## Installation

### 前置要求
- Python 3.10+
- pip

### 步驟
1. Clone 倉庫: 
   ```bash
   git clone https://github.com/yourusername/cosmetic-analyzer.git
   cd cosmetic-analyzer
   ```

2. 安裝依賴:
   ```bash
   pip install -r requirements.txt
   ```

3. 設置環境變數 (創建 `.env` 文件):
   ```bash
   GROQ_API_KEY_1=your_api_key_here
   GROQ_API_KEY_2=your_api_key_here
   GROQ_API_KEY_3=your_api_key_here
   ```

4. 構建 FAISS 索引:
   ```bash
   python build_index.py
   ```

5. 運行應用:
   ```bash
   python app.py
   ```

## Usage
See [Usage Instructions](USAGE.md) for detailed guidance.

## Project Structure
```
├── app.py                 # Gradio 前端應用
├── build_index.py         # 建立 FAISS 索引
├── requirements.txt       # 依賴列表
├── rag/                   # RAG 引擎
│   ├── chain.py          # LLM 鏈
│   ├── graph.py          # LangGraph 工作流
│   ├── retriever.py      # FAISS 向量庫
│   ├── enricher.py       # LLM 補充引擎
│   ├── ingestor.py       # 數據載入
│   └── validator.py      # 格式驗證
├── data/                  # 數據文件
│   ├── ingredients.json   # 官方成分數據庫
│   └── pending_ingredients.json  # 待驗證的新成分
└── faiss_index/          # 向量索引文件
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.