#!/usr/bin/env python3
"""
Script de teste para a função de comparação de títulos de seções
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluator import SurGEvaluator

def main():
    # Configurações
    device = "0"
    survey_path = "data/surveysMatScience.json"
    corpus_path = "data/corpusMatScience.json"
    api_key = ""
    
    # Inicializa o avaliador
    print("Inicializando avaliador...")
    evaluator = SurGEvaluator(
        device=device,
        survey_path=survey_path,
        corpus_path=corpus_path,
        using_openai=True,
        api_key=api_key
    )
    
    # Testa com um survey existente
    print("\nTestando comparação de títulos de seções...")
    
    # ID do survey para testar
    survey_id = 0
    passage_path = "./baselines/ID/output/0/0.md"
    
    if os.path.exists(passage_path):
        # Executa a avaliação com a opção de comparar títulos
        result = evaluator.single_eval(
            survey_id=survey_id,
            passage_path=passage_path,
            eval_list=["Compare_Section_Titles"]
        )
        print("\n✓ Teste concluído com sucesso!")
    else:
        print(f"✗ Arquivo não encontrado: {passage_path}")
        print("  Tente ajustar o survey_id e passage_path")

if __name__ == "__main__":
    main()
