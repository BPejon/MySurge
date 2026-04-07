# Resumo das Mudanças Implementadas

## Data: 1 de Abril de 2026

### Objetivo
Implementar uma função para comparar os títulos das seções de um artigo gerado pela LLM com o artigo do Golden Truth, utilizando embeddings e medidas de distância coseno.

---

## Arquivos Modificados

### 1. **src/evaluator.py**

#### Importações Adicionadas
```python
import numpy as np
from scipy.spatial.distance import cdist
```

#### Nova Função: `compare_section_titles()`

**Localização**: Classe `SurGEvaluator` (linhas ~77-175)

**Funcionalidades**:
- Extrai todos os títulos de seções do artigo Golden Truth
- Extrai todos os títulos de seções do artigo gerado pela LLM
- Gera embeddings vetoriais para cada título usando `FlagModel`
- Calcula matriz de similaridade coseno
- Converte para matriz de distância (1 - similaridade)
- Para cada título do GT, encontra o mais similar da LLM
- Ordena resultados por distância (menor = mais similar)
- Exibe resultados formatados em tabela

**Retorno**: Dicionário com:
- `gt_titles`: Lista de títulos do Golden Truth
- `llm_titles`: Lista de títulos da LLM
- `comparisons`: Lista ordenada de comparações com distância e similaridade

#### Integração no `single_eval()`

A função é chamada automaticamente quando:
- `"Compare_Section_Titles"` está em `eval_list`
- `"ALL"` está em `eval_list`

---

## Arquivos Criados

### 1. **test_section_titles.py**
Script de teste básico para validar a funcionalidade.

**Como usar**:
```bash
python test_section_titles.py
```

### 2. **examples_section_titles.py**
Arquivo com 4 exemplos práticos de uso:
1. **Exemplo 1**: Comparação básica direta
2. **Exemplo 2**: Usando `eval_list` no `single_eval()`
3. **Exemplo 3**: Analisando múltiplos surveys
4. **Exemplo 4**: Análise detalhada com categorização de qualidade

**Como usar**:
```bash
python examples_section_titles.py
```

### 3. **SECTION_TITLES_COMPARISON_DOC.md**
Documentação completa com:
- Explicação da implementação
- Assinatura de função
- Parâmetros e retorno
- Exemplos de uso
- Interpretação de métricas
- Dependências necessárias

---

## Técnica Utilizada

### Embeddings
- **Modelo**: `BAAI/bge-large-en-v1.5` (FlagModel)
- **Dimensionidade**: 1024 (padrão do modelo)
- **Normalização**: Vetores normalizados para cálculo de similaridade coseno

### Métrica de Distância
- **Tipo**: Distância Coseno (1 - similaridade)
- **Intervalo**: 0 a 2
  - 0 = títulos idênticos
  - 1 = títulos ortogonais (sem relação)
  - 2 = títulos opostos

### Similaridade
- **Tipo**: Produto escalar de vetores normalizados
- **Intervalo**: -1 a 1
  - 1 = máxima similaridade
  - 0 = ortogonalidade
  - -1 = oposição

---

## Fluxo da Função

```
┌─────────────────────────────────┐
│ compare_section_titles()        │
└──────────────┬──────────────────┘
               │
               ├─→ Extrai títulos GT
               ├─→ Extrai títulos LLM
               │
               ├─→ Verifica se há títulos
               │   (retorna se vazio)
               │
               ├─→ Carrega FlagModel
               │   (se não carregado)
               │
               ├─→ Gera embeddings GT
               ├─→ Gera embeddings LLM
               │
               ├─→ Normaliza embeddings
               ├─→ Calcula similaridade
               ├─→ Converte para distância
               │
               ├─→ Encontra correspondências
               │   (each GT → closest LLM)
               │
               ├─→ Ordena por distância
               │
               ├─→ Exibe tabela formatada
               │
               └─→ Retorna dicionário com resultados
```

---

## Exemplos de Output

### Entrada
```
Golden Truth Titles:
  1. Introduction
  2. Related Work
  3. Proposed Method
  4. Experiments
  5. Conclusion

LLM Generated Titles:
  1. Introduction to the Problem
  2. Literature Review
  3. Our Approach
  4. Experimental Results
  5. Future Work
  6. References
```

### Output
```
================================================================================
RESULTADOS DA COMPARAÇÃO (Ordenados por menor distância):
================================================================================
Distância    Similaridade   Título GT                          Título LLM
--------------------------------------------------------------------------------
0.0847       0.9153         Introduction                       Introduction to the Problem
0.1234       0.8766         Related Work                       Literature Review
0.1692       0.8308         Proposed Method                    Our Approach
0.2145       0.7855         Experiments                        Experimental Results
0.3456       0.6544         Conclusion                         Future Work
================================================================================
```

---

## Validações Implementadas

1. ✓ Verificação se `survey_id` existe em `survey_map`
2. ✓ Filtragem de seções com < 100 caracteres de conteúdo
3. ✓ Exclusão de títulos "root" e "Abstract:"
4. ✓ Verificação se há títulos para comparar
5. ✓ Carregamento lazy do FlagModel
6. ✓ Normalização correta de embeddings
7. ✓ Formatação segura de strings na exibição

---

## Dependências

Todas as dependências já existem em `requirements.txt`:
- `numpy` ≥ 1.20
- `scipy` ≥ 1.5
- `FlagEmbedding` ≥ 1.0
- `sentence-transformers` ≥ 2.0

---

## Performance

### Tempo de Execução Estimado
- Carregamento do modelo: ~30-60 segundos (primeira vez)
- Embedding de 10 títulos: ~1-2 segundos
- Cálculo de distâncias: <1 segundo

### Uso de Memória
- Modelo FlagModel: ~2-3 GB
- Embeddings por título: ~4 KB
- Total para 100 títulos: <1 MB adicional

---

## Casos de Uso

1. **Avaliação de Qualidade**: Verificar se a LLM preserva a estrutura do GT
2. **Análise de Similaridade**: Identificar quais seções foram bem geradas
3. **Detecção de Problemas**: Encontrar seções com estruturas muito diferentes
4. **Comparação Batch**: Analisar múltiplos surveys simultaneamente

---

## Próximos Passos (Opcionais)

1. Adicionar cache de embeddings para melhorar performance em runs repetidas
2. Implementar visualização gráfica das correspondências
3. Adicionar análise hierárquica de subsecções
4. Integrar com sistema de logging estruturado
5. Adicionar threshold customizável para filtro de qualidade

---

## Teste Rápido

Para fazer um teste rápido usando a CLI:

```python
python3 -c "
import sys
sys.path.insert(0, 'src')
from evaluator import SurGEvaluator
import markdownParser

# Setup
evaluator = SurGEvaluator(
    device='0',
    survey_path='data/surveysMatScience.json',
    corpus_path='data/corpusMatScience.json',
    using_openai=False
)

# Test
psg_node = markdownParser.parse_markdown('./baselines/ID/output/26/0.md')
result = evaluator.compare_section_titles(26, psg_node)
print('✓ Teste concluído!')
"
```

---

## Notas Técnicas

1. **Por que normalizar embeddings?**
   - Garante que o cálculo de similaridade coseno seja entre -1 e 1
   - Melhora a interpretabilidade dos resultados

2. **Por que usar 1 - similaridade como distância?**
   - Transforma valores de [-1, 1] em [0, 2]
   - 0 = idêntico, 1 = ortogonal
   - Mais intuitivo para "distância"

3. **Por que um embedding por título?**
   - Simplicidade de implementação
   - FlagModel foi treinado para fazer isso bem
   - Reduz complexidade computacional

---

## Contato e Suporte

Para dúvidas ou problemas, consulte:
- SECTION_TITLES_COMPARISON_DOC.md
- examples_section_titles.py
- test_section_titles.py
