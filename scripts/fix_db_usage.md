# 列出 DB 所有成分，確認哪些是 LLM-generated
python scripts/fix_db.py list

# 修正單一成分
python scripts/fix_db.py fix --names Aqua

# 批次修正多個成分
python scripts/fix_db.py fix-batch --names Aqua Bakuchiol

# 一鍵修正所有 LLM-generated 的成分
python scripts/fix_db.py fix-llm

# 刪除指定成分
python scripts/fix_db.py remove --names Aqua