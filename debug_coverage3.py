"""
Encontrar qual citação está faltando
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

# Criar mapa docid -> title
docid2title = {}
for c in corpus:
    docid2title[int(c['doc_id'])] = c['Title']

# Parse refs
refs = markdownParser.parse_refs(gt_path)

# Mapear refs para docids
title2docid = {}
for c in corpus:
    normalized_title = normalize_string(c['Title'])
    title2docid[normalized_title] = int(c['doc_id'])

refid2docid = {}
for refid, ref_title in refs.items():
    normalized = normalize_string(ref_title)
    if normalized in title2docid:
        refid2docid[refid] = title2docid[normalized]
    else:
        refid2docid[refid] = ref_title

# Agora encontrar qual doc_id não foi mapeado
gen_docids = set()
for v in refid2docid.values():
    if isinstance(v, int):
        gen_docids.add(v)

survey_docids = set(survey['all_cites'])

missing = survey_docids - gen_docids
found_as_string = refid2docid.values()
found_as_string = [v for v in found_as_string if isinstance(v, str)]

print(f"Survey citations: {len(survey['all_cites'])}")
print(f"Generated (mapped) citations: {len(gen_docids)}")
print(f"Found as strings (not found): {len(found_as_string)}")
print(f"\nMissing doc IDs: {sorted(missing)}")
print(f"\nFound as strings: {found_as_string[:5]}")

print(f"\n\nMissing Document Titles:")
for doc_id in sorted(missing):
    if doc_id in docid2title:
        title = docid2title[doc_id]
        print(f"  Doc {doc_id}: {title}")
