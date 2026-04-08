"""
Investigação detalhada do problema de Coverage
"""

import json
import sys
sys.path.insert(0, './src')

import markdownParser
from evaluator import normalize_string

# Carregar os dados
with open('./data/surveysMatScience.json', 'r', encoding='utf-8') as f:
    surveys = json.load(f)

with open('./data/corpusMatScience.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Survey 0
survey = surveys[0]
gt_path = './data/artigoGT.md'

# Parse refs
refs = markdownParser.parse_refs(gt_path)

print(f"Total refs from file: {len(refs)}")
print(f"Total cite IDs in survey: {len(survey['all_cites'])}")
print(f"Total docs in corpus: {len(corpus)}")

# Vamos examinar algumas refs problemáticas
problematics = []
for refid, ref_title in refs.items():
    if len(ref_title) < 10 or 'org/' in ref_title or ref_title.startswith('J ') or ref_title.startswith('Petrophysics'):
        problematics.append((refid, ref_title))

print(f"\n\nProblematic refs (short or incomplete titles):")
for refid, title in problematics[:20]:
    print(f"  [{refid}]: '{title}'")

# Vamos olhar para estas referências no arquivo original
print(f"\n\nOpen the markdown file to check some problematic refs:")
with open(gt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Procurar [3], [4], [22], [27], etc
problematic_ids = [3, 4, 22, 27, 50, 58, 77, 78, 79, 81, 83]

for ref_id in problematic_ids[:5]:
    # Encontra a linha com [ref_id]
    for i, line in enumerate(lines):
        if f'[{ref_id}]' in line:
            # Pega as próximas 3 linhas
            print(f"\n\nRef [{ref_id}] (line {i+1}):")
            for j in range(i, min(i+4, len(lines))):
                print(f"  {lines[j].rstrip()[:80]}")
            break
