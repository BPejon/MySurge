#!/usr/bin/env python3
"""
Script integral de comparação detalhada de seções.
Testa a funcionalidade completa com os arquivos reais.
"""

import sys
import os

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import evaluator
import markdownParser
import structureFuncs
from FlagEmbedding import FlagModel
import numpy as np

def main():
    """Executa a comparação completa"""
    
    # Caminhos
    llm_path = "/home/breno/Documentos/onealSurge/baselines/ID/output/0/0.md"
    gt_path = "/home/breno/Documentos/onealSurge/dataMatScience/artigoGT.md"
    output_path = "/home/breno/Documentos/onealSurge/outputs/detailed_sections_comparison.txt"
    
    print("="*100)
    print("COMPARAÇÃO COMPLETA DE SEÇÕES SIMILARES")
    print("="*100)
    
    # Cria instância do evaluador (sem API key pois vamos usar apenas a função de comparação)
    try:
        print("\n[1/3] Inicializando SurGEvaluator...")
        eval_obj = evaluator.SurGEvaluator(
            survey_path="/home/breno/Documentos/onealSurge/dataMatScience/surveysMatScience.json",
            corpus_path="/home/breno/Documentos/onealSurge/dataMatScience/corpusMatScience.json",
            using_openai=False
        )
        print("✓ SurGEvaluator inicializado")
    except Exception as e:
        print(f"✗ Erro ao inicializar: {e}")
        return False
    
    # Executa a comparação
    try:
        print("\n[2/3] Executando comparação detalhada de seções...")
        comparison_results = eval_obj.compare_sections_detailed(
            llm_md_path=llm_path,
            gt_md_path=gt_path,
            threshold=0.85
        )
        print(f"✓ Comparação concluída: {len(comparison_results)} matches encontrados")
    except Exception as e:
        print(f"✗ Erro na comparação: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Exporta os resultados
    try:
        print("\n[3/3] Exportando resultados para arquivo...")
        eval_obj.export_sections_comparison(
            comparison_data=comparison_results,
            output_path=output_path
        )
        print(f"✓ Arquivo gerado: {output_path}")
    except Exception as e:
        print(f"✗ Erro ao exportar: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Imprime sumário
    print("\n" + "="*100)
    print("RESUMO DOS RESULTADOS")
    print("="*100)
    
    if comparison_results:
        print(f"\nTotal de matches (similaridade >= 0.85): {len(comparison_results)}\n")
        print(f"{'Seção':<40} {'Distância':<15} {'Similaridade':<15} {'BERTScore':<15} {'ROUGE-L':<15}")
        print("-"*100)
        
        for item in comparison_results[:10]:  # Mostra apenas os 10 primeiros
            print(f"{item['section_gt_name'][:40]:<40} {item['distance']:<15.4f} {item['similarity']:<15.4f} {item['bertscore_f1']:<15.4f} {item['rouge_l']:<15.4f}")
        
        if len(comparison_results) > 10:
            print(f"... ({len(comparison_results) - 10} mais items)")
    else:
        print("\nNenhum match encontrado com similaridade >= 0.85")
    
    print("\n" + "="*100)
    print("✓ SUCESSO! Resultados exportados para:")
    print(f"  {output_path}")
    print("="*100)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
