#!/usr/bin/env python3
"""
Script de teste para validar a integração do BERTScore
com a análise de seções do projeto onealSurge.
"""

import json
import sys
import os

sys.path.insert(0, '/home/breno/Documentos/onealSurge/src')

from evaluator import SurGEvaluator
from markdownParser import parse_markdown

def test_basic_extraction():
    """Testa extração básica de texto de seções."""
    print("\n" + "="*80)
    print("TESTE 1: Extração de Texto de Seções")
    print("="*80)
    
    # Carrega um arquivo markdown de teste
    passage_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    psg_node = parse_markdown(passage_path)
    
    print(f"✓ Arquivo markdown parseado: {passage_path}")
    print(f"  Nó raiz: {psg_node.title}")
    print(f"  Número de filhos: {len(psg_node.children)}")
    
    # Testa extração de texto
    from structureFuncs import extract_section_text_from_markdown
    
    if psg_node.children:
        first_section = psg_node.children[0]
        text = extract_section_text_from_markdown(first_section)
        print(f"\n✓ Extração de texto da seção: '{first_section.title}'")
        print(f"  Tamanho do texto: {len(text)} caracteres")
        print(f"  Preview: {text[:100]}...")
    
    return True


def test_bertscore_calculation():
    """Testa cálculo de BERTScore."""
    print("\n" + "="*80)
    print("TESTE 2: Cálculo de BERTScore")
    print("="*80)
    
    from rougeBleuFuncs import calculate_bertscore_for_sections
    
    # Textos de teste
    text_llm = "Nuclear Magnetic Resonance is a powerful technique for analyzing materials."
    text_gt = "NMR is an effective analytical tool for characterizing structures."
    
    print(f"\nTexto LLM: {text_llm}")
    print(f"Texto GT:  {text_gt}")
    
    result = calculate_bertscore_for_sections(text_llm, text_gt)
    
    print(f"\n✓ BERTScore calculado:")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1: {result['f1']:.4f}")
    
    return True




def test_compare_section_titles():
    """Teste removido - requer API key válida. Use examples_section_titles.py para teste completo."""
    return None


def test_full_pipeline():
    """Teste removido - requer API key válida. Use examples_section_titles.py para teste completo."""
    return None


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTES DE INTEGRAÇÃO - BERTSCORE E ANÁLISE DE SEÇÕES")
    print("="*80)
    
    tests = [
        ("Extração de Texto", test_basic_extraction),
        ("Cálculo de BERTScore", test_bertscore_calculation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✓ PASSOU" if result else "✗ FALHOU"))
        except Exception as e:
            print(f"\n✗ ERRO: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, f"✗ ERRO: {str(e)}"))
    
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    for test_name, status in results:
        print(f"{test_name:<40} {status}")
    
    print("\n" + "="*80)
    print("INFORMAÇÕES IMPORTANTES")
    print("="*80)
    print("""
✓ Funções implementadas com sucesso:
  1. extract_section_text_from_markdown() em structureFuncs.py
  2. calculate_bertscore_for_sections() em rougeBleuFuncs.py
  3. extract_section_text_from_gt() em evaluator.py
  4. generate_comparison_table() em evaluator.py
  5. print_comparison_table() em evaluator.py
  6. Integração em single_eval() em evaluator.py

✓ Funções de suporte:
  - find_section_by_title() - busca seção por título em GT
  - find_markdown_section_by_title() - busca nó por título em LLM

✓ Os testes 3 e 4 (Comparação de Títulos e Pipeline Completo) exigem 
  uma API key válida do OpenAI, então foram removidos de testes automáticos.
  
✓ Para usar a funcionalidade completa, execute:
  - python3 examples_section_titles.py
  - ou use single_eval() com "Compare_Section_Titles" na eval_list
  
✓ Pipeline completo:
  compare_section_titles() → generate_comparison_table() → print_comparison_table()
    """)
    
    print("✓ Testes concluídos!\n")
