import json

with open('paper/notebooks/05_brand_relevance.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][11]
print("".join(cell.get('source', [])))
if 'outputs' in cell and cell['outputs']:
    for out in cell['outputs']:
        if 'text' in out:
            print("".join(out['text']))
