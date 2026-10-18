import json

with open('paper/notebooks/01_cleaning.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = "".join(cell.get('source', []))
    lines = src.split('\n')
    header = lines[0] if lines else ""
    print(f"--- Cell {i} ({ctype}) ---: {header[:100]}")
    if ctype == 'code':
        print("\n".join(lines[:12]))
