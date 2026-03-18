# Fix Database Usage Guide

This guide explains how to use the `fix_db.py` script to manage and update the ingredient database.

## Prerequisites
- Ensure the database and FAISS index are properly set up.
- Set up environment variables in `.env` file (GROQ_API_KEY_1, etc.).

## Commands

### List all ingredients in the database and check which are LLM-generated
```bash
python scripts/fix_db.py list
```

### Fix a single ingredient
```bash
python scripts/fix_db.py fix --names Aqua
```

### Batch fix multiple ingredients
```bash
python scripts/fix_db.py fix-batch --names Aqua Bakuchiol
```

### Fix all LLM-generated ingredients in one go
```bash
python scripts/fix_db.py fix-llm
```

### Remove specified ingredients
```bash
python scripts/fix_db.py remove --names Aqua
```

## Notes
- The script will automatically rebuild the FAISS index after modifications.
- Use with caution when removing ingredients, as this action cannot be undone.