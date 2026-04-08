"""
Debug script para entender o problema de Coverage
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

# Criar um mapa de título normalizado para doc_id
title2docid = {}
for c in corpus:
    normalized_title = normalize_string(c['Title'])
    title2docid[normalized_title] = int(c['doc_id'])
    # Debug: print the title mapping
    # print(f"Corpus Doc {c['doc_id']}: {c['Title'][:60]}... -> normalized: {normalized_title}")

# Pegar o primeiro survey para testar
survey = surveys[0]
survey_id = int(survey['survey_id'])
gt_path = './data/artigoGT.md'

print(f"Survey ID: {survey_id}")
print(f"GT file: {gt_path}")
print(f"All cites from GT: {survey['all_cites'][:20]}")  # Primeiras 20

# Parse das refs do arquivo
refs = markdownParser.parse_refs(gt_path)
print(f"\n\nNumber of refs parsed from file: {len(refs)}")

# Mapear refs para docids
refid2docid = {}
for refid, ref_title in refs.items():
    normalized = normalize_string(ref_title)
    if normalized in title2docid:
        refid2docid[refid] = title2docid[normalized]
        print(f"✓ Reference [{refid}]: '{ref_title[:50]}...' -> Doc ID {title2docid[normalized]}")
    else:
        refid2docid[refid] = ref_title
        print(f"✗ Reference [{refid}]: '{ref_title[:50]}...' -> NOT FOUND (stored as string)")

# Agora calcular coverage manualmente
target_cites = survey['all_cites']
gen_cites = list(refid2docid.values())

print(f"\n\n==== COVERAGE CALCULATION ====")
print(f"Target cites count: {len(target_cites)}")
print(f"Generated cites count: {len(gen_cites)}")
print(f"\nTarget cites (IDs): {target_cites[:20]}")
print(f"\nGenerated cites (IDs + strings): {gen_cites[:20]}")

hit = 0
miss = []
for c in target_cites:
    if c in gen_cites:
        hit += 1
    else:
        miss.append(c)

print(f"\n\nHits: {hit}")
print(f"Coverage: {hit / len(target_cites)}")
print(f"Misses (first 10): {miss[:10]}")

# Vamos investigar os misses
print(f"\n\n==== INVESTIGATING MISSES ====")
for doc_id in miss[:5]:
    # Encontra o título no corpus
    doc_title = None
    for c in corpus:
        if int(c['doc_id']) == doc_id:
            doc_title = c['Title']
            break
    
    if doc_title:
        normalized_doc = normalize_string(doc_title)
        print(f"\nMissing Doc {doc_id}: {doc_title[:60]}...")
        print(f"  Normalized title: {normalized_doc}")
        
        # Procura similaridade nos títulos extractados
        for refid, ref_title in list(refs.items())[:5]:
            normalized_ref = normalize_string(ref_title)
            if normalized_doc == normalized_ref:
                print(f"  FOUND MATCH with ref [{refid}]: {ref_title[:60]}...")
