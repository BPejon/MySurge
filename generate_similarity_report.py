#!/usr/bin/env python3
"""
Script para gerar relatório visual dos pares de seções mais similares.
Exibe os TOP pares ordenados por similaridade.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_similarity_report(csv_path, json_path, top_n=50):
    """
    Gera relatório visual dos pares mais similares.
    """
    
    print("\n" + "="*150)
    print("RELATÓRIO DE SIMILARIDADE DE SEÇÕES - TOP {} PARES MAIS SIMILARES".format(top_n))
    print("="*150)
    
    # Lê CSV
    df = pd.read_csv(csv_path)
    
    # Lê JSON para contexto
    with open(json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    stats = result.get('statistics_details', {})
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   • Similaridade Máxima: {stats.get('max_similarity', 0):.4f}")
    print(f"   • Similaridade Média: {stats.get('mean_similarity', 0):.4f}")
    print(f"   • Similaridade Mínima: {stats.get('min_similarity', 0):.4f}")
    print(f"   • Total de pares similares encontrados: {len(df)}")
    
    # Ordena por similaridade decrescente
    df_sorted = df.sort_values('Similaridade', ascending=False)
    
    # Mostra TOP N
    top_df = df_sorted.head(top_n)
    
    print(f"\n" + "="*150)
    print(f"TOP {top_n} PARES COM MAIOR SIMILARIDADE")
    print("="*150)
    
    # Cabeçalho formatado
    print(f"{'Rank':<6} {'GT Título':<40} {'LLM Título':<40} {'Sim':<8} {'Nível GT':<12} {'Nível LLM':<12} {'Tamanho':<10}")
    print("-"*150)
    
    # Dados
    for idx, (_, row) in enumerate(top_df.iterrows(), 1):
        gt_title = str(row['GT_Título'])[:38]
        llm_title = str(row['LLM_Título'])[:38]
        similarity = float(row['Similaridade'])
        tamanho = f"{int(row['Tamanho_GT'])}/{int(row['Tamanho_LLM'])}"
        
        # Colore por similaridade
        if similarity >= 0.90:
            marker = "⭐⭐⭐"
        elif similarity >= 0.80:
            marker = "⭐⭐"
        else:
            marker = "⭐"
        
        print(f"{idx:<6} {gt_title:<40} {llm_title:<40} {similarity:<8.4f} {row['Nível_GT']:<12} {row['Nível_LLM']:<12} {tamanho:<10} {marker}")
    
    print("="*150)
    
    # Análise por nível hierárquico
    print(f"\n📈 DISTRIBUIÇÃO POR NÍVEL HIERÁRQUICO:")
    nivel_counts = df['Nível_GT'].value_counts().sort_index()
    for nivel, count in nivel_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {nivel}: {count:4d} pares ({percentage:5.1f}%)")
    
    # Análise de similaridade por faixa
    print(f"\n📊 DISTRIBUIÇÃO POR FAIXA DE SIMILARIDADE:")
    ranges = [(0.90, 1.00), (0.80, 0.90), (0.75, 0.80), (0.70, 0.75)]
    for min_sim, max_sim in ranges:
        count = len(df[(df['Similaridade'] >= min_sim) & (df['Similaridade'] < max_sim)])
        percentage = (count / len(df)) * 100
        label = f"[{min_sim:.2f}, {max_sim:.2f})"
        bar = "█" * int(percentage / 2)
        print(f"   {label}: {count:4d} pares ({percentage:5.1f}%) {bar}")
    
    # Top 10 por título
    print(f"\n🏆 TOP 10 TÍTULOS GT COM MAIOR SIMILARIDADE MÉDIA:")
    top_titles_gt = df.groupby('GT_Título')['Similaridade'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
    for idx, (title, row) in enumerate(top_titles_gt.iterrows(), 1):
        print(f"   {idx:2d}. {title:<50} | Média: {row['mean']:.4f} | Ocorrências: {int(row['count'])}")
    
    print(f"\n🏆 TOP 10 TÍTULOS LLM COM MAIOR SIMILARIDADE MÉDIA:")
    top_titles_llm = df.groupby('LLM_Título')['Similaridade'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
    for idx, (title, row) in enumerate(top_titles_llm.iterrows(), 1):
        print(f"   {idx:2d}. {title:<50} | Média: {row['mean']:.4f} | Ocorrências: {int(row['count'])}")
    
    # Exportar TOP para arquivo de texto
    output_txt = "outputs/top_similar_pairs.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*150 + "\n")
        f.write("RELATÓRIO DE SIMILARIDADE DE SEÇÕES - TOP 100 PARES MAIS SIMILARES\n")
        f.write("="*150 + "\n\n")
        
        f.write(f"ESTATÍSTICAS GERAIS:\n")
        f.write(f"   • Similaridade Máxima: {stats.get('max_similarity', 0):.4f}\n")
        f.write(f"   • Similaridade Média: {stats.get('mean_similarity', 0):.4f}\n")
        f.write(f"   • Similaridade Mínima: {stats.get('min_similarity', 0):.4f}\n")
        f.write(f"   • Total de pares similares encontrados: {len(df)}\n\n")
        
        f.write("="*150 + "\n")
        f.write("TOP 100 PARES COM MAIOR SIMILARIDADE\n")
        f.write("="*150 + "\n\n")
        
        top_100 = df_sorted.head(100)
        for idx, (_, row) in enumerate(top_100.iterrows(), 1):
            f.write(f"{idx:3d}. GT: {row['GT_Título']:<50} | LLM: {row['LLM_Título']:<50}\n")
            f.write(f"     Similaridade: {row['Similaridade']:.4f} | Níveis: {row['Nível_GT']} -> {row['Nível_LLM']} | Tamanho: {int(row['Tamanho_GT'])} vs {int(row['Tamanho_LLM'])}\n")
            f.write(f"     Caminho GT: {row['Caminho_GT']}\n")
            f.write(f"     Caminho LLM: {row['Caminho_LLM']}\n\n")
    
    print(f"\n✓ Relatório salvo em: {output_txt}")
    print("\n" + "="*150 + "\n")


if __name__ == '__main__':
    csv_path = "outputs/content_similarity_table.csv"
    json_path = "outputs/content_similarity_result.json"
    
    if os.path.exists(csv_path) and os.path.exists(json_path):
        generate_similarity_report(csv_path, json_path, top_n=50)
    else:
        print(f"❌ Arquivos não encontrados:")
        print(f"   CSV: {csv_path} (existe: {os.path.exists(csv_path)})")
        print(f"   JSON: {json_path} (existe: {os.path.exists(json_path)})")
        sys.exit(1)
