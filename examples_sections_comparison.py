#!/usr/bin/env python3
"""
EXEMPLO DE USO: Comparação Detalhada de Seções com BERTScore

Este exemplo mostra como usar a nova funcionalidade para comparar seções
e subseções entre artigos LLM e Ground Truth com análise detalhada de conteúdo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluator import SurGEvaluator

def example_1_basic_comparison():
    """
    Exemplo 1: Comparação básica entre dois arquivos markdown
    This is the simplest way to use the comparison functionality.
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: Comparação Básica")
    print("="*80 + "\n")
    
    # Caminhos dos arquivos
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    
    # Inicializa o evaluador
    evaluator = SurGEvaluator(
        survey_path="/home/breno/Documentos/onealSurge/dataMatScience/surveysMatScience.json",
        corpus_path="/home/breno/Documentos/onealSurge/dataMatScience/corpusMatScience.json",
        using_openai=False
    )
    
    # Executa a comparação detalhada
    comparison_results = evaluator.compare_sections_detailed(
        llm_md_path=llm_path,
        gt_md_path=gt_path,
        threshold=0.85
    )
    
    print(f"\nTotal de matches encontrados: {len(comparison_results)}\n")
    
    # Exibe sumário dos primeiros 5 matches
    print("Primeiros 5 matches:")
    print(f"{'GT Section':<40} {'Sim':<10} {'BERTScore':<12} {'ROUGE-L':<10}")
    print("-"*72)
    
    for i, item in enumerate(comparison_results[:5], 1):
        print(f"{item['section_gt_name'][:40]:<40} "
              f"{item['similarity']:<10.4f} "
              f"{item['bertscore_f1']:<12.4f} "
              f"{item['rouge_l']:<10.4f}")
    
    if len(comparison_results) > 5:
        print(f"... e mais {len(comparison_results) - 5} matches")


def example_2_export_results():
    """
    Exemplo 2: Comparação e exportação dos resultados em arquivo texto
    This shows how to export the comparison results to a formatted text file.
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: Comparação e Exportação")
    print("="*80 + "\n")
    
    # Caminhos
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    output_path = "/home/breno/Documentos/onealSurge/outputs/example_comparison.txt"
    
    # Inicializa o evaluador
    evaluator = SurGEvaluator(
        survey_path="/home/breno/Documentos/onealSurge/dataMatScience/surveysMatScience.json",
        corpus_path="/home/breno/Documentos/onealSurge/dataMatScience/corpusMatScience.json",
        using_openai=False
    )
    
    # Executa a comparação
    comparison_results = evaluator.compare_sections_detailed(
        llm_md_path=llm_path,
        gt_md_path=gt_path,
        threshold=0.85
    )
    
    # Exporta os resultados
    evaluator.export_sections_comparison(
        comparison_data=comparison_results,
        output_path=output_path
    )
    
    print(f"Resultados exportados para: {output_path}")


def example_3_custom_threshold():
    """
    Exemplo 3: Uso de diferentes thresholds de similaridade
    You can adjust the threshold to get more or fewer matches.
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: Diferentes Thresholds")
    print("="*80 + "\n")
    
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    
    evaluator = SurGEvaluator(
        survey_path="/home/breno/Documentos/onealSurge/dataMatScience/surveysMatScience.json",
        corpus_path="/home/breno/Documentos/onealSurge/dataMatScience/corpusMatScience.json",
        using_openai=False
    )
    
    # Testa com diferentes thresholds
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for threshold in thresholds:
        results = evaluator.compare_sections_detailed(
            llm_md_path=llm_path,
            gt_md_path=gt_path,
            threshold=threshold
        )
        print(f"Threshold {threshold}: {len(results)} matches encontrados")


def example_4_analyze_single_match():
    """
    Exemplo 4: Análise detalhada de um match individual
    This shows how to access and analyze individual comparison results.
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: Análise Detalhada de Um Match")
    print("="*80 + "\n")
    
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    
    evaluator = SurGEvaluator(
        survey_path="/home/breno/Documentos/onealSurge/dataMatScience/surveysMatScience.json",
        corpus_path="/home/breno/Documentos/onealSurge/dataMatScience/corpusMatScience.json",
        using_openai=False
    )
    
    results = evaluator.compare_sections_detailed(
        llm_md_path=llm_path,
        gt_md_path=gt_path,
        threshold=0.85
    )
    
    if results:
        # Analisa o primeiro match
        first_match = results[0]
        
        print(f"Seção GT: {first_match['section_gt_name']}")
        print(f"Seção LLM: {first_match['section_llm_name']}")
        print(f"\nMétricas:")
        print(f"  Distância: {first_match['distance']:.4f}")
        print(f"  Similaridade: {first_match['similarity']:.4f}")
        print(f"  BERTScore F1: {first_match['bertscore_f1']:.4f}")
        print(f"  ROUGE-L: {first_match['rouge_l']:.4f}")
        
        print(f"\nTexto GT (primeiros 200 chars):")
        print(f"  {first_match['text_gt'][:200]}...")
        
        print(f"\nTexto LLM (primeiros 200 chars):")
        print(f"  {first_match['text_llm'][:200]}...")


# Funções de utilidade para entender melhor os dados
def print_available_metrics():
    """Imprime as métricas disponíveis em um resultado de comparação"""
    print("\n" + "="*80)
    print("MÉTRICAS DISPONÍVEIS")
    print("="*80 + "\n")
    
    metrics_info = {
        "section_llm_name": "Nome da seção no artigo LLM",
        "section_gt_name": "Nome da seção no artigo GT",
        "distance": "Distância euclidiana normalizada (0-1, menor = mais similar)",
        "similarity": "Similaridade coseno (0-1, maior = mais similar)",
        "bertscore_f1": "BERTScore F1 score (0-1, mede similaridade semântica com BERT)",
        "rouge_l": "ROUGE-L score (0-1, mede overlapping de longest common subsequence)",
        "text_llm": "Texto completo da seção no artigo LLM",
        "text_gt": "Texto completo da seção no artigo GT"
    }
    
    for metric, description in metrics_info.items():
        print(f"  • {metric}")
        print(f"    {description}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exemplos de uso da funcionalidade de comparação detalhada")
    parser.add_argument("--example", choices=["1", "2", "3", "4", "metrics", "all"], 
                       default="all", help="Qual exemplo executar")
    
    args = parser.parse_args()
    
    if args.example in ["1", "all"]:
        example_1_basic_comparison()
    if args.example in ["2", "all"]:
        example_2_export_results()
    if args.example in ["3", "all"]:
        example_3_custom_threshold()
    if args.example in ["4", "all"]:
        example_4_analyze_single_match()
    if args.example in ["metrics", "all"]:
        print_available_metrics()
    
    print("\n" + "="*80)
    print("✓ Exemplos completos!")
    print("="*80 + "\n")
