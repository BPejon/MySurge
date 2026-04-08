"""
Implementar fuzzy matching para melhorar ainda mais o coverage
"""

import json
import sys
sys.path.insert(0, './src')

import markdownParser
from evaluator import normalize_string
from difflib import SequenceMatcher

def similarity(a, b):
    """Calcula similaridade entre duas strings"""
    return SequenceMatcher(None, a, b).ratio()

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

# Encontrar matches mais robustos com fuzzy matching
target_cites = survey['all_cites']
gen_docids = set()
found_as_string = {}

for v in refid2docid.values():
    if isinstance(v, int):
        gen_docids.add(v)
    else:
        # É uma string - título não foi encontrado
        found_as_string[normalize_string(v)] = v

survey_docids = set(target_cites)
missing = survey_docids - gen_docids

print(f"Missing document IDs: {len(missing)}")
print(f"Fuzzy matching for missing docs...\n")

fuzzy_matches = 0
threshold = 0.75

for doc_id in sorted(missing)[:10]:  # Primeiros 10 para teste
    target_title = normalize_string(docid2title[doc_id])
    
    # Procura melhor match entre as strings encontradas
    best_match = None
    best_score = 0
    
    for found_norm, found_orig in found_as_string.items():
        score = similarity(target_title, found_norm)
        if score > best_score and score > threshold:
            best_score = score
            best_match = found_orig
    
    if best_match:
        fuzzy_matches += 1
        print(f"Doc {doc_id}: MATCH with score {best_score:.2f}")
        print(f"  Target: {docid2title[doc_id][:60]}...")
        print(f"  Found:  {best_match[:60]}...")
        print()
    else:
        print(f"Doc {doc_id}: NO MATCH")
        print(f"  Target: {docid2title[doc_id][:60]}...")
        # Show best attempt even if below threshold
        best_attempt = max(
            [(found_norm, found_orig, similarity(target_title, found_norm)) 
             for found_norm, found_orig in found_as_string.items()],
            key=lambda x: x[2],
            default=(None, None, 0)
        )
        if best_attempt[0]:
            print(f"  Best attempt (score {best_attempt[2]:.2f}): {best_attempt[1][:60]}...")
        print()

print(f"\nFuzzy matches found: {fuzzy_matches}/10")
print(f"This could improve coverage to approximately {(283 + fuzzy_matches)/314:.2%}")
