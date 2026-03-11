import json
from rag.retriever import build_index

with open("data/ingredients.json", "r", encoding="utf-8") as f:
    ingredients = json.load(f)

build_index(ingredients)