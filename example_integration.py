#!/usr/bin/env python3
"""
Exemplo de integração: Como usar extract_references_improved() no seu projeto.

Este script demonstra como integrar a nova extração de referências
melhorada com busca em corpus no seu pipeline de avaliação.
"""

import sys
sys.path.insert(0, 'src')

from informationFuncs import extract_references_improved
import json

# Exemplo 1: Extração simples de um arquivo markdown
print("=" * 80)
print("EXEMPLO 1: Extração de Referências de um Arquivo")
print("=" * 80)

# Simular leitura do arquivo 0.md
with open("baselines/ID/output/0/0.md", 'r', encoding='utf-8') as f:
    content = f.read()

# Encontrar a seção de referências
references_start = content.find("## References")
if references_start != -1:
    references_section = content[references_start:]
    
    print(f"\nExtendo referências de {len(references_section)} caracteres da seção...")
    
    # Usar a função melhorada
    results = extract_references_improved(
        references_section,
        corpus_path="dataMatScience/corpusMatScience.json"
    )
    
    print(f"\nEncontradas {len(results)} referências:\n")
    for ref_num, content_result in results:
        print(f"  [{ref_num}] -> {content_result}")

# Exemplo 2: Análise de sucesso/falha
print("\n" + "=" * 80)
print("EXEMPLO 2: Análise de Sucessos (encontrou no corpus)")
print("=" * 80)

found_in_corpus = []
not_found = []

for ref_num, content_result in results:
    if ':' in content_result:
        parts = content_result.split(':', 1)
        if len(parts) == 2:
            _, value = parts
            # Se value é um número, encontrou no corpus
            if value.isdigit():
                found_in_corpus.append((ref_num, int(value)))
            else:
                not_found.append((ref_num, value))

print(f"\n✓ Encontradas no corpus: {len(found_in_corpus)}")
for ref_num, doc_id in found_in_corpus[:5]:  # mostra apenas os 5 primeiros
    print(f"  [{ref_num}] -> doc_id={doc_id}")

print(f"\n✗ Não encontradas no corpus: {len(not_found)}")
for ref_num, title in not_found[:3]:  # mostra apenas os 3 primeiros
    print(f"  [{ref_num}] -> {title[:60]}...")

# Exemplo 3: Estatísticas
print("\n" + "=" * 80)
print("EXEMPLO 3: Estatísticas")
print("=" * 80)

total = len(results)
success = len(found_in_corpus)
coverage = (success / total * 100) if total > 0 else 0

print(f"\nTotal de referências: {total}")
print(f"Encontradas: {success} ({coverage:.1f}%)")
print(f"Não encontradas: {len(not_found)} ({100-coverage:.1f}%)")

# Exemplo 4: Como usar no seu avaliador
print("\n" + "=" * 80)
print("EXEMPLO 4: Código para integrar no src/evaluator.py")
print("=" * 80)

integration_code = '''
# No arquivo src/evaluator.py, substitua a função que extrai referências:

# ANTES (linha ~170):
def extract_cites_with_subtitle_and_sentence(psg_node: MarkdownNode):
    res = []
    text = "\\n".join(psg_node.content)
    tmp_res = extract_references(text)  # ← ANTIGA
    ...

# DEPOIS:
def extract_cites_with_subtitle_and_sentence(psg_node: MarkdownNode):
    res = []
    text = "\\n".join(psg_node.content)
    tmp_res = extract_references_improved(text, "dataMatScience/corpusMatScience.json")  # ← NOVA
    ...

# A função extract_references_improved retorna tuples (ref_num, content)
# onde content pode ser:
#   - "1:9" (encontrou doc_id=9)
#   - "1:Título do artigo" (não encontrou, retorna título)
'''

print(integration_code)

print("\n" + "=" * 80)
print("FIM DO EXEMPLO")
print("=" * 80)
