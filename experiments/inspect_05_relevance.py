import json

with open('paper/notebooks/05_brand_relevance.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [7, 11, 13, 14, 16, 18, 19]:
    cell = nb['cells'][idx]
    print(f"==================== CELL {idx} ({cell.get('cell_type')}) ====================")
    print("".join(cell.get('source', [])))
    if 'outputs' in cell and cell['outputs']:
        print("--- OUTPUTS ---")
        for out in cell['outputs']:
            if 'text' in out:
                print("".join(out['text']))
            elif 'data' in out and 'text/plain' in out['data']:
                print("".join(out['data']['text/plain']))
