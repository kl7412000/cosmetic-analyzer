#!/usr/bin/env python
from rag.graph import analyze_online

results = analyze_online('Fragrance')
print(f'Results: {len(results)}')
for r in results:
    name = r.get("_display_name") or r.get("ingredient") or "Unknown"
    conf = r.get("confidence")
    print(f"  - {name} ({conf})")
