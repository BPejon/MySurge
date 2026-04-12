#!/usr/bin/env python3
"""
Script de teste para a funcionalidade de comparação detalhada de seções.
Testa com os arquivos reais: artigoGT.md e 0.md
"""

import sys
import os

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import evaluator
import markdownParser
import structureFuncs

def test_sections_comparison():
    """Testa a comparação detalhada de seções"""
    
    # Caminhos dos arquivos
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    output_path = "/home/breno/Documentos/onealSurge/outputs/detailed_sections_comparison.txt"
    
    print("="*100)
    print("TESTE: Comparação Detalhada de Seções")
    print("="*100)
    print(f"\nArquivos:")
    print(f"  LLM: {llm_path}")
    print(f"  GT: {gt_path}")
    print(f"  Output: {output_path}")
    
    # Verifica se os arquivos existem
    if not os.path.exists(llm_path):
        print(f"ERRO: Arquivo LLM não encontrado: {llm_path}")
        return False
    if not os.path.exists(gt_path):
        print(f"ERRO: Arquivo GT não encontrado: {gt_path}")
        return False
    
    print("\n[1/4] Parsando arquivos markdown...")
    try:
        llm_node = markdownParser.parse_markdown(llm_path)
        gt_node = markdownParser.parse_markdown(gt_path)
        print("✓ Arquivos parseados com sucesso")
    except Exception as e:
        print(f"✗ Erro ao parsear: {e}")
        return False
    
    print("\n[2/4] Extraindo títulos...")
    try:
        llm_titles = structureFuncs.get_title_list(llm_node)
        gt_titles = structureFuncs.get_title_list(gt_node)
        print(f"✓ Títulos LLM: {len(llm_titles)}")
        print(f"✓ Títulos GT: {len(gt_titles)}")
        
        # Mostra primeiros 5 títulos de cada
        print(f"\n  Primeiros 5 títulos LLM:")
        for i, title in enumerate(llm_titles[:5], 1):
            print(f"    {i}. {title[:60]}")
        print(f"\n  Primeiros 5 títulos GT:")
        for i, title in enumerate(gt_titles[:5], 1):
            print(f"    {i}. {title[:60]}")
    except Exception as e:
        print(f"✗ Erro ao extrair títulos: {e}")
        return False
    
    print("\n[3/4] Testando extração de texto de seções...")
    try:
        # Encontra e extrai texto de uma seção
        if llm_node.children:
            first_section = llm_node.children[0]
            text = structureFuncs.extract_section_text(first_section)
            print(f"✓ Texto extraído (primeiras 200 chars): {text[:200]}")
        else:
            print("! Nenhuma seção encontrada no nó LLM")
    except Exception as e:
        print(f"✗ Erro ao extrair texto: {e}")
        return False
    
    print("\n[4/4] Testando cálculo de métricas de similaridade...")
    try:
        # Testa com dois textos simples
        text1 = "Nuclear Magnetic Resonance is a powerful analytical technique"
        text2 = "NMR is an important tool for materials analysis"
        
        metrics = structureFuncs.calculate_content_metrics(text1, text2)
        print(f"✓ Métricas calculadas:")
        print(f"  - BERTScore F1: {metrics['bertscore_f1']:.4f}")
        print(f"  - ROUGE-L: {metrics['rouge_l']:.4f}")
    except Exception as e:
        print(f"✗ Erro ao calcular métricas: {e}")
        print("  (Isso é esperado se BERTScore não está instalado)")
        return False
    
    print("\n" + "="*100)
    print("✓ TESTES BÁSICOS PASSARAM!")
    print("="*100)
    
    return True


if __name__ == "__main__":
    success = test_sections_comparison()
    sys.exit(0 if success else 1)
