#!/usr/bin/env python3
"""
Exemplo de uso da função de comparação de títulos de seções
"""

import sys
import os

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluator import SurGEvaluator
import markdownParser

def exemplo_basico():
    """Exemplo básico de uso da comparação de títulos"""
    
    print("=" * 80)
    print("EXEMPLO 1: Comparação Básica de Títulos de Seções")
    print("=" * 80)
    
    # Configurações
    survey_path = "data/surveysMatScience.json"
    corpus_path = "data/corpusMatScience.json"
    
    # Inicializa o avaliador (sem OpenAI para este exemplo)
    evaluator = SurGEvaluator(
        device="0",
        survey_path=survey_path,
        corpus_path=corpus_path,
        using_openai=False
    )
    
    # Teste com um arquivo existente
    survey_id = 26
    passage_path = "./baselines/ID/output/26/0.md"
    
    if os.path.exists(passage_path):
        # Faz o parsing do markdown
        psg_node = markdownParser.parse_markdown(passage_path)
        
        # Chama a função de comparação diretamente
        result = evaluator.compare_section_titles(survey_id, psg_node)
        
        print(f"\nResumo dos resultados:")
        print(f"- Títulos do GT: {len(result['gt_titles'])}")
        print(f"- Títulos da LLM: {len(result['llm_titles'])}")
        print(f"- Comparações (ordenadas por similaridade): {len(result['comparisons'])}")
        
        if result['comparisons']:
            print(f"\nTítulo mais similar:")
            best = result['comparisons'][0]
            print(f"  GT: {best['gt_title']}")
            print(f"  LLM: {best['llm_title']}")
            print(f"  Distância: {best['distance']:.4f}")
            print(f"  Similaridade: {best['similarity']:.4f}")
    else:
        print(f"Arquivo não encontrado: {passage_path}")


def exemplo_com_eval_list():
    """Exemplo usando eval_list no single_eval"""
    
    print("\n" + "=" * 80)
    print("EXEMPLO 2: Usando eval_list no single_eval()")
    print("=" * 80)
    
    survey_path = "data/surveysMatScience.json"
    corpus_path = "data/corpusMatScience.json"
    
    evaluator = SurGEvaluator(
        device="0",
        survey_path=survey_path,
        corpus_path=corpus_path,
        using_openai=False
    )
    
    survey_id = 26
    passage_path = "./baselines/ID/output/26/0.md"
    
    if os.path.exists(passage_path):
        # Chama single_eval com Compare_Section_Titles na eval_list
        result = evaluator.single_eval(
            survey_id=survey_id,
            passage_path=passage_path,
            eval_list=["Compare_Section_Titles", "SH-Recall"]
        )
        print("\n✓ Avaliação completa realizada!")
    else:
        print(f"Arquivo não encontrado: {passage_path}")


def exemplo_multiplos_surveys():
    """Exemplo comparando múltiplos surveys"""
    
    print("\n" + "=" * 80)
    print("EXEMPLO 3: Analisando Múltiplos Surveys")
    print("=" * 80)
    
    survey_path = "data/surveysMatScience.json"
    corpus_path = "data/corpusMatScience.json"
    
    evaluator = SurGEvaluator(
        device="0",
        survey_path=survey_path,
        corpus_path=corpus_path,
        using_openai=False
    )
    
    # Diretório com múltiplos surveys
    passage_dir = "./baselines/ID/output"
    
    if os.path.exists(passage_dir):
        survey_dirs = [
            d for d in os.listdir(passage_dir)
            if os.path.isdir(os.path.join(passage_dir, d))
        ]
        
        print(f"\nEncontrados {len(survey_dirs)} survey(s)")
        
        for survey_id in survey_dirs[:2]:  # Processa apenas os 2 primeiros
            survey_path_full = os.path.join(passage_dir, survey_id)
            md_files = [f for f in os.listdir(survey_path_full) if f.endswith('.md')]
            
            if md_files:
                md_file = os.path.join(survey_path_full, md_files[0])
                print(f"\nProcessando Survey {survey_id}...")
                
                psg_node = markdownParser.parse_markdown(md_file)
                result = evaluator.compare_section_titles(int(survey_id), psg_node)
    else:
        print(f"Diretório não encontrado: {passage_dir}")


def exemplo_analise_detalhada():
    """Exemplo com análise detalhada dos resultados"""
    
    print("\n" + "=" * 80)
    print("EXEMPLO 4: Análise Detalhada dos Resultados")
    print("=" * 80)
    
    survey_path = "data/surveysMatScience.json"
    corpus_path = "data/corpusMatScience.json"
    
    evaluator = SurGEvaluator(
        device="0",
        survey_path=survey_path,
        corpus_path=corpus_path,
        using_openai=False
    )
    
    survey_id = 26
    passage_path = "./baselines/ID/output/26/0.md"
    
    if os.path.exists(passage_path):
        psg_node = markdownParser.parse_markdown(passage_path)
        result = evaluator.compare_section_titles(survey_id, psg_node)
        
        # Análise dos resultados
        if result['comparisons']:
            print("\n" + "-" * 80)
            print("ANÁLISE DETALHADA:")
            print("-" * 80)
            
            distances = [c['distance'] for c in result['comparisons']]
            similarities = [c['similarity'] for c in result['comparisons']]
            
            print(f"\nDistâncias:")
            print(f"  Mínima: {min(distances):.4f}")
            print(f"  Máxima: {max(distances):.4f}")
            print(f"  Média: {sum(distances) / len(distances):.4f}")
            
            print(f"\nSimilaridades:")
            print(f"  Mínima: {min(similarities):.4f}")
            print(f"  Máxima: {max(similarities):.4f}")
            print(f"  Média: {sum(similarities) / len(similarities):.4f}")
            
            # Categoriza por qualidade de correspondência
            excelente = [c for c in result['comparisons'] if c['distance'] < 0.15]
            bom = [c for c in result['comparisons'] if 0.15 <= c['distance'] < 0.30]
            moderado = [c for c in result['comparisons'] if 0.30 <= c['distance'] < 0.50]
            ruim = [c for c in result['comparisons'] if c['distance'] >= 0.50]
            
            print(f"\nQualidade das Correspondências:")
            print(f"  ✓ Excelente (dist < 0.15): {len(excelente)}")
            print(f"  ✓ Bom (0.15 ≤ dist < 0.30): {len(bom)}")
            print(f"  ○ Moderado (0.30 ≤ dist < 0.50): {len(moderado)}")
            print(f"  ✗ Ruim (dist ≥ 0.50): {len(ruim)}")
    else:
        print(f"Arquivo não encontrado: {passage_path}")


if __name__ == "__main__":
    try:
        # Executa os exemplos
        exemplo_basico()
        exemplo_com_eval_list()
        exemplo_multiplos_surveys()
        exemplo_analise_detalhada()
        
        print("\n" + "=" * 80)
        print("✓ Todos os exemplos foram executados com sucesso!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
