#!/usr/bin/env python3
"""
Script para testar a extração melhorada de referências.
Valida o funcionamento das novas funções para o formato journal.
"""

import sys
sys.path.insert(0, 'src')

from informationFuncs import (
    extract_references_improved,
    is_journal_format,
    extract_title_from_reference,
    find_title_in_corpus,
    load_corpus_index
)

# Teste 1: Referência com formato journal
print("=" * 80)
print("TESTE 1: Extração de Referência com Formato Journal")
print("=" * 80)

reference_text = """[1] Karem Al-Garadi, Ammar El-Hussein, Mahmoud Elsayed, P. Connolly, Mohamed Mahmoud, M. Johns, and A. Adebayo. A rock core wettability index using NMR T2 measurements. *Journal of Petroleum Science and Engineering*, 208:109386, 2021."""

print(f"\nReferência original:\n{reference_text}\n")

# Teste 2: Detectar formato
print("-" * 80)
print("TESTE 2: Detectar Formato Journal")
print("-" * 80)
is_journal = is_journal_format(reference_text)
print(f"É formato journal? {is_journal}")
assert is_journal == True, "Deveria detectar como formato journal!"

# Teste 3: Extrair título
print("\n" + "-" * 80)
print("TESTE 3: Extrair Título")
print("-" * 80)
title = extract_title_from_reference(reference_text)
print(f"Título extraído: {title}")
expected_title = "A rock core wettability index using NMR T2 measurements"
assert title == expected_title, f"Título incorreto! Esperava: {expected_title}"

# Teste 4: Buscar no corpus
print("\n" + "-" * 80)
print("TESTE 4: Buscar Título no Corpus")
print("-" * 80)

# Carrega índice
index = load_corpus_index("dataMatScience/corpusMatScience.json")
print(f"Corpus carregado com {len(index)} títulos")

doc_id = find_title_in_corpus(title, "dataMatScience/corpusMatScience.json")
print(f"Doc ID encontrado: {doc_id}")
assert doc_id == 9, f"Deveria encontrar doc_id=9, encontrou: {doc_id}"

# Teste 5: Extração completa
print("\n" + "-" * 80)
print("TESTE 5: Extração Completa com extract_references_improved")
print("-" * 80)

full_text = f"""
## Seção de Referências

{reference_text}

[2] Mohammad Elsayed et al. Some other paper. *Other Journal*, 2021.
"""

results = extract_references_improved(full_text, "dataMatScience/corpusMatScience.json")
print(f"\nResultados extraídos:")
for ref_num, content in results:
    print(f"  [{ref_num}] -> {content}")

# Validar resultados
assert len(results) == 2, f"Deveria ter 2 referências, encontrou {len(results)}"
assert results[0] == (1, "1:9"), f"Primeira referência incorreta: {results[0]}"
print(f"\n✓ SUCESSO: Primeira referência retornou '1:9' (encontrou doc_id=9)")

print("\n" + "=" * 80)
print("TODOS OS TESTES PASSARAM! ✓")
print("=" * 80)
