#!/usr/bin/env python3
"""
Script PRINCIPAL para análise completa de similaridade de seções.
Combina extração de seções, cálculo de embeddings e relatório visual.

Uso:
    python analyze_section_similarity.py <survey_id> <passage_path>
    python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md"
"""

import sys
import os
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluator import SurGEvaluator
import markdownParser


def main():
    parser = argparse.ArgumentParser(
        description='Analisa similaridade de seções entre GT e LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md"
  python analyze_section_similarity.py 1 "outputs/llm_article.md" --threshold 0.8
  python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md" --top 100
        """
    )
    
    parser.add_argument('survey_id', type=int, help='ID do survey no Ground Truth')
    parser.add_argument('passage_path', type=str, help='Caminho para arquivo markdown do artigo LLM')
    parser.add_argument('--threshold', type=float, default=0.70, 
                       help='Threshold mínimo de similaridade (default: 0.70)')
    parser.add_argument('--top', type=int, default=50,
                       help='Número de top pares a exibir (default: 50)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Diretório para salvar resultados (default: outputs)')
    
    args = parser.parse_args()
    
    # Validações
    if not os.path.exists(args.passage_path):
        print(f"❌ Erro: Arquivo não encontrado: {args.passage_path}")
        sys.exit(1)
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print("ANÁLISE DE SIMILARIDADE DE SEÇÕES")
    print("="*100)
    print(f"\n📋 Configuração:")
    print(f"   Survey ID: {args.survey_id}")
    print(f"   Arquivo: {args.passage_path}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Top resultados: {args.top}")
    print(f"   Output: {args.output_dir}/")
    
    # Step 1: Inicializar avaliador
    print(f"\n[1/4] Inicializando avaliador...")
    try:
        evaluator = SurGEvaluator(
            device="0",
            survey_path="data/surveys.json",
            corpus_path="data/corpus.json",
            flag_model_path='BAAI/bge-large-en-v1.5',
            using_openai=False
        )
        print("✓ Avaliador inicializado")
    except Exception as e:
        print(f"❌ Erro ao inicializar: {str(e)}")
        sys.exit(1)
    
    # Step 2: Parse do markdown
    print(f"\n[2/4] Parseando arquivo markdown...")
    try:
        psg_node = markdownParser.parse_markdown(args.passage_path)
        print("✓ Markdown parseado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao parsear: {str(e)}")
        sys.exit(1)
    
    # Step 3: Executar comparação
    print(f"\n[3/4] Comparando seções...")
    try:
        result = evaluator.compare_section_content_hierarchical(
            args.survey_id, 
            psg_node, 
            similarity_threshold=args.threshold
        )
        
        gt_sections = result.get('gt_sections', [])
        llm_sections = result.get('llm_sections', [])
        similar_pairs = result.get('similar_pairs', [])
        comparison_df = result.get('comparison_df')
        
        print(f"✓ Comparação concluída")
        print(f"   • Seções GT: {len(gt_sections)}")
        print(f"   • Seções LLM: {len(llm_sections)}")
        print(f"   • Pares similares: {len(similar_pairs)}")
        
    except Exception as e:
        print(f"❌ Erro ao comparar: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Gerar relatório e salvar resultados
    print(f"\n[4/4] Gerando relatório...")
    
    # Mostrar tabela na tela
    print("\n" + "="*100)
    print(f"TOP {min(args.top, len(similar_pairs))} PARES COM MAIOR SIMILARIDADE")
    print("="*100)
    
    if comparison_df is not None and not comparison_df.empty:
        # Ordena por similaridade
        df_sorted = comparison_df.sort_values('Similaridade', ascending=False)
        top_df = df_sorted.head(args.top)
        
        # Cabeçalho
        print(f"{'#':<5} {'GT_Título':<40} {'LLM_Título':<40} {'Sim':<8} {'Níveis':<20} {'Tamanho':<10}")
        print("-"*100)
        
        # Dados
        for idx, (_, row) in enumerate(top_df.iterrows(), 1):
            gt_title = str(row['GT_Título'])[:38]
            llm_title = str(row['LLM_Título'])[:38]
            similarity = float(row['Similaridade'])
            levels = f"{row['Nível_GT']}/{row['Nível_LLM']}"
            tamanho = f"{int(row['Tamanho_GT'])}/{int(row['Tamanho_LLM'])}"
            
            print(f"{idx:<5} {gt_title:<40} {llm_title:<40} {similarity:<8.4f} {levels:<20} {tamanho:<10}")
        
        print("="*100)
        
        # Estatísticas
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   • Similaridade Máxima: {comparison_df['Similaridade'].max():.4f}")
        print(f"   • Similaridade Média: {comparison_df['Similaridade'].mean():.4f}")
        print(f"   • Similaridade Mínima: {comparison_df['Similaridade'].min():.4f}")
        print(f"   • Mediana: {comparison_df['Similaridade'].median():.4f}")
        print(f"   • Desvio Padrão: {comparison_df['Similaridade'].std():.4f}")
        
        # Salvar resultados
        csv_path = f"{args.output_dir}/content_similarity_table_{args.survey_id}.csv"
        json_path = f"{args.output_dir}/content_similarity_result_{args.survey_id}.json"
        txt_path = f"{args.output_dir}/content_similarity_report_{args.survey_id}.txt"
        
        # CSV
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # JSON
        result_to_save = {
            'survey_id': args.survey_id,
            'passages_file': args.passage_path,
            'threshold': args.threshold,
            'statistics': {
                'gt_sections_count': len(gt_sections),
                'llm_sections_count': len(llm_sections),
                'similar_pairs_count': len(similar_pairs),
            },
            'statistics_details': {
                'max_similarity': float(comparison_df['Similaridade'].max()),
                'mean_similarity': float(comparison_df['Similaridade'].mean()),
                'median_similarity': float(comparison_df['Similaridade'].median()),
                'min_similarity': float(comparison_df['Similaridade'].min()),
                'std_similarity': float(comparison_df['Similaridade'].std()),
            },
            'top_pairs': [
                {
                    'gt_title': p['gt_title'],
                    'llm_title': p['llm_title'],
                    'similarity': float(p['similarity']),
                    'gt_level': p['gt_level'],
                    'llm_level': p['llm_level'],
                    'gt_path': p['gt_path'],
                    'llm_path': p['llm_path'],
                }
                for p in similar_pairs[:100]  # Top 100
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_to_save, f, ensure_ascii=False, indent=2)
        
        # TXT Relatório
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"ANÁLISE DE SIMILARIDADE DE SEÇÕES - Survey {args.survey_id}\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Arquivo: {args.passage_path}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ESTATÍSTICAS:\n")
            f.write(f"  Seções GT: {len(gt_sections)}\n")
            f.write(f"  Seções LLM: {len(llm_sections)}\n")
            f.write(f"  Pares Similares: {len(similar_pairs)}\n")
            f.write(f"  Similaridade Máxima: {comparison_df['Similaridade'].max():.4f}\n")
            f.write(f"  Similaridade Média: {comparison_df['Similaridade'].mean():.4f}\n")
            f.write(f"  Similaridade Mínima: {comparison_df['Similaridade'].min():.4f}\n\n")
            
            f.write("="*100 + "\n")
            f.write(f"TOP {min(100, len(comparison_df))} PARES\n")
            f.write("="*100 + "\n\n")
            
            top_100 = comparison_df.sort_values('Similaridade', ascending=False).head(100)
            for idx, (_, row) in enumerate(top_100.iterrows(), 1):
                f.write(f"{idx:3d}. GT: {row['GT_Título']}\n")
                f.write(f"      LLM: {row['LLM_Título']}\n")
                f.write(f"      Similaridade: {row['Similaridade']:.4f}\n")
                f.write(f"      Níveis: {row['Nível_GT']} -> {row['Nível_LLM']}\n")
                f.write(f"      Caminhos: {row['Caminho_GT']} / {row['Caminho_LLM']}\n\n")
        
        print(f"\n✅ RESULTADOS SALVOS:")
        print(f"   • CSV: {csv_path}")
        print(f"   • JSON: {json_path}")
        print(f"   • TXT: {txt_path}")
        
    else:
        print("❌ Nenhum par similar encontrado com o threshold especificado")
    
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
