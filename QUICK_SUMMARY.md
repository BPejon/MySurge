# 🎯 Resumo da Implementação - Análise de Similaridade de Seções

## ✅ O Que Foi Implementado

### 1. **Funções de Extração Hierárquica** (`evaluator.py`)

```python
# Extrai seções do artigo LLM de forma hierárquica
extract_sections_hierarchical_from_llm(markdown_node, level=0, parent_path=None)

# Extrai seções do Ground Truth de forma hierárquica
extract_sections_hierarchical_from_gt(survey_id)
```

Ambas retornam:
```
[
    {'title': str, 'level': int, 'content': str, 'path_titles': list},
    ...
]
```

### 2. **Função Principal de Comparação** (`evaluator.py`)

```python
compare_section_content_hierarchical(survey_id, psg_node, similarity_threshold=0.75)
```

Retorna:
```python
{
    'gt_sections': list,
    'llm_sections': list,
    'similarity_matrix': np.ndarray,    # Matriz M×N de similaridades
    'similar_pairs': list,              # Pares ordenados por similaridade
    'comparison_df': pd.DataFrame       # Tabela para exportação
}
```

### 3. **Função de Visualização** (`evaluator.py`)

```python
print_similar_pairs_table(comparison_result)
```

Imprime tabela formatada dos pares similares.

### 4. **Script CLI Principal** ⭐

`analyze_section_similarity.py` - Script de linha de comando com:
- Validação de argumentos
- Interface amigável
- Exportação automática (CSV, JSON, TXT)
- Estatísticas detalhadas

---

## 🚀 Como Usar (Tl;DR)

### Comando Básico
```bash
python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md"
```

### Com Opções
```bash
python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md" \
    --threshold 0.80 --top 50 --output-dir "results"
```

### Opções Disponíveis
- `--threshold`: Similaridade mínima (default: 0.70)
- `--top`: Número de top pares a mostrar (default: 50)
- `--output-dir`: Diretório para salvar (default: outputs)

---

## 📊 Saídas Geradas

Para cada análise, são criados 3 arquivos:

| Arquivo | Conteúdo | Uso |
|---------|----------|-----|
| `content_similarity_table_*.csv` | Tabela completa | Análise em Excel/Python |
| `content_similarity_result_*.json` | Dados estruturados | Integração com outros scripts |
| `content_similarity_report_*.txt` | Relatório formatado | Leitura humana |

---

## 📈 Exemplo de Resultado

```
TOP 30 PARES COM MAIOR SIMILARIDADE
====================================================================================================
#  GT_Título                              LLM_Título                          Sim     Níveis    Tamanho
----------------------------------------------------------------------------------------------------
1  Transformer with Convolution           5.1 Image Classification            0.9102  L4/L3     425/574
2  Challenges                             5.1 Image Classification            0.8970  L3/L3     396/574
3  Generic Object Detection               5.2 Object Detection                0.8950  L1/L3     1866/810
...

ESTATÍSTICAS:
   • Similaridade Máxima: 0.9102
   • Similaridade Média: 0.8281
   • Mediana: 0.8223
   • Desvio Padrão: 0.0238
```

---

## 🔧 Configuração Técnica

### Dependências Instaladas
- ✅ pandas
- ✅ numpy
- ✅ scipy
- ✅ sentence-transformers
- ✅ FlagEmbedding

### Modelo Utilizado
- **FlagEmbedding**: BAAI/bge-large-en-v1.5
- **Métrica**: Similaridade Coseno
- **Embedding Size**: 1024 dimensões

### Processo
1. Extrai seções hierarquicamente (GT e LLM)
2. Codifica conteúdo em embeddings com FlagModel
3. Normaliza embeddings L2
4. Calcula matriz de similaridade: `similarity = dot(gt_emb, llm_emb.T)`
5. Filtra pares acima do threshold
6. Ordena por similaridade decrescente

---

## 💡 Exemplos Práticos

### Exemplo 1: Encontrar Pares Muito Similares
```bash
python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md" \
    --threshold 0.85 --top 30
```
→ Apenas pares com similaridade >= 0.85

### Exemplo 2: Análise Completa com Threshold Baixo
```bash
python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md" \
    --threshold 0.70 --top 100
```
→ Ver os 100 melhores pares (podem ser ~4000+)

### Exemplo 3: Survey Diferente
```bash
python analyze_section_similarity.py 1 "baselines/Autosurvey/output/1/survey.md" \
    --threshold 0.75 --output-dir "outputs/survey1"
```
→ Análise de survey ID 1 com resultados em diretório custom

---

## 📝 Arquivos Criados/Modificados

### Novos Arquivos
- ✅ `analyze_section_similarity.py` - Script CLI principal
- ✅ `test_content_similarity.py` - Script de teste
- ✅ `generate_similarity_report.py` - Gerador de relatório
- ✅ `SIMILARITY_ANALYSIS_README.md` - Documentação completa
- ✅ `QUICK_SUMMARY.md` - Este arquivo

### Modificações em `src/evaluator.py`
- ✅ `extract_sections_hierarchical_from_llm()` - Nova função
- ✅ `extract_sections_hierarchical_from_gt()` - Nova função (corrigida)
- ✅ `compare_section_content_hierarchical()` - Função melhorada
- ✅ `print_similar_pairs_table()` - Nova função de visualização

---

## 🎓 O Que Cada Função Faz

### `extract_sections_hierarchical_from_llm()`
- **Input**: MarkdownNode (raiz do artigo parseado)
- **Output**: Lista de seções com `{title, level, content, path_titles}`
- **Uso**: Extrair estrutura do artigo LLM

### `extract_sections_hierarchical_from_gt()`
- **Input**: survey_id (ID do survey no Ground Truth)
- **Output**: Mesma estrutura do LLM
- **Uso**: Extrair estrutura do Ground Truth

### `compare_section_content_hierarchical()`
- **Input**: survey_id, psg_node, threshold
- **Output**: Dict com matriz de similaridade e pares similares
- **Uso**: Calcular similaridades entre seções

### `print_similar_pairs_table()`
- **Input**: Resultado de compare_section_content_hierarchical()
- **Output**: Tabela formatada na tela
- **Uso**: Visualizar resultados

---

## 📊 Interpretação de Resultados

### Similaridade Coseno

| Score | Interpretação | Exemplo |
|-------|--------------|---------|
| **0.90+** | Muito similar | "Transformer Architecture" vs "Architecture of Transformers" |
| **0.80-0.89** | Bastante similar | "Convolution Layers" vs "Convolutional Components" |
| **0.70-0.79** | Moderadamente similar | "Vision Tasks" vs "Computer Vision Applications" |
| **< 0.70** | Pouco similar | Seções sobre tópicos diferentes |

### Níveis Hierárquicos

- **Nível 1**: Seção principal (ex: "Introduction")
- **Nível 2**: Subsecção (ex: "1.1 Background")
- **Nível 3+**: Subsubsecção, etc.

---

## 🐛 Se der erro...

### Erro: "Arquivo não encontrado"
```
Verifique o caminho do arquivo markdown
ls -la "baselines/Autosurvey/output/26/"
```

### Erro: "ModuleNotFoundError"
```
Instale dependências:
pip install pandas numpy scipy sentence-transformers FlagEmbedding
```

### Lentidão
Use threshold mais alto para reduzir comparações:
```bash
python analyze_section_similarity.py 26 "..." --threshold 0.85
```

---

## 📚 Arquivos de Documentação

1. **SIMILARITY_ANALYSIS_README.md** - Documentação completa
2. **QUICK_SUMMARY.md** - Este arquivo (resumo rápido)
3. **outputs/content_similarity_report_*.txt** - Relatório de execução
4. **outputs/content_similarity_result_*.json** - Dados estruturados

---

## ✨ Resumo

**Implementado:**
✅ Extração hierárquica de seções (GT e LLM)  
✅ Cálculo de embeddings semânticos  
✅ Matriz de similaridade coseno  
✅ Identificação de pares similares  
✅ Exportação (CSV, JSON, TXT)  
✅ Script CLI completo com opções  
✅ Documentação detalhada  

**Pronto para usar:**
```bash
python analyze_section_similarity.py 26 "baselines/Autosurvey/output/26/A Survey on Visual Transformer_.md"
```

---

**Status**: ✅ **IMPLEMENTADO E TESTADO**

Qualquer dúvida, consulte `SIMILARITY_ANALYSIS_README.md`
