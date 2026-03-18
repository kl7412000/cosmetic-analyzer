---
title: Cosmetic Ingredient Analyzer
emoji: 🧴
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "6.9.0"
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
cosmetic-analyzer/
├── app.py  # 主應用程式檔案，負責運行整個化妝品分析器應用
├── build_index.py  # 建構FAISS向量索引的腳本，用於資料檢索
├── README.md  # 專案說明文件，包含專案概述和安裝指南
├── requirements.txt  # Python依賴包列表，指定所需的第三方庫
├── runtime.txt  # 運行時環境配置檔案，指定Python版本（如用於Heroku部署）
├── USAGE.md  # 使用說明文件，詳細說明如何使用應用程式
├── __pycache__/  # Python編譯快取目錄，自動生成，無需手動編輯
├── data/
│   ├── ingredients.json  # 化妝品成分資料庫，儲存已知成分資訊
│   └── pending_ingredients.json  # 待處理的成分資料，等待驗證或整合
├── faiss_index/
│   └── index.faiss  # FAISS向量索引檔案，用於高效的相似性搜尋
├── rag/
│   ├── __init__.py  # RAG（Retrieval-Augmented Generation）模組初始化檔案
│   ├── chain.py  # RAG鏈處理模組，管理檢索和生成鏈
│   ├── enricher.py  # 資料豐富化模組，增強成分資訊
│   ├── graph.py  # 圖形處理模組，可能用於知識圖譜或關係圖
│   ├── groq_client.py  # Groq API客戶端，處理與Groq語言模型的互動
│   ├── ingestor.py  # 資料攝取模組，負責將新資料加入系統
│   ├── ocr.py  # OCR（光學字元辨識）模組，用於從圖片提取文字
│   ├── offline_graph.py  # 離線圖形處理模組，處理無網路環境下的圖形操作
│   ├── retriever.py  # 檢索模組，從索引中檢索相關資料
│   ├── updater.py  # 資料更新模組，處理資料庫更新
│   ├── validator.py  # 資料驗證模組，檢查資料正確性
│   └── __pycache__/  # RAG模組的Python編譯快取目錄
├── scraper/
│   ├── __init__.py  # 刮取模組初始化檔案
│   ├── cosing.py  # COSING資料庫刮取模組，從歐洲化妝品成分資料庫獲取資料
│   ├── inci_decoder.py  # INCI解碼器模組，解析化妝品成分名稱
│   ├── paulas_choice.py  # Paula's Choice刮取模組，從Paula's Choice網站獲取評價資料
│   └── __pycache__/  # 刮取模組的Python編譯快取目錄
├── scripts/
│   ├── fix_db_usage.md  # 修復資料庫使用說明文件
│   └── fix_db.py  # 修復資料庫的腳本，處理資料庫問題
└── tests/
    ├── __init__.py  # 測試模組初始化檔案
    └── test_core.py  # 核心功能測試檔案，包含單元測試
```

## Testing
Run the test suite using pytest:
```bash
pip install pytest
pytest tests/
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.