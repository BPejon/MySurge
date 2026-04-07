# Documentação: Comparação de Títulos de Seções

## Resumo da Implementação

Foi implementada uma nova funcionalidade no arquivo `src/evaluator.py` para comparar os títulos das seções entre um artigo gerado pela LLM e o artigo do Golden Truth (GT).

## Função Principal: `compare_section_titles()`

### Localização
- **Arquivo**: `src/evaluator.py`
- **Classe**: `SurGEvaluator`

### O que faz

A função realiza as seguintes operações:

1. **Extração de Títulos do GT**: Obtém os títulos das seções do artigo Golden Truth a partir da estrutura armazenada em `survey_map`
   
2. **Extração de Títulos da LLM**: Extrai os títulos das seções do arquivo markdown gerado pela LLM usando `structureFuncs.get_title_list()`

3. **Geração de Embeddings**: Cria representações vetoriais dos títulos usando o modelo `FlagModel` (BAAI/bge-large-en-v1.5)
   - Normaliza os embeddings para cálculo de similaridade coseno

4. **Cálculo de Distâncias**: Para cada título do GT, encontra o título da LLM mais próximo usando distância coseno
   - **Distância**: 1 - similaridade coseno (entre 0 e 2)
   - **Similaridade**: Produto escalar dos embeddings normalizados (entre -1 e 1)

5. **Impressão de Resultados**: Exibe os resultados formatados em ordem crescente de distância:
   - Título do GT
   - Título correspondente da LLM
   - Distância entre eles
   - Score de similaridade

### Assinatura

```python
def compare_section_titles(self, survey_id: int, psg_node: MarkdownNode) -> dict
```

### Parâmetros

- **`survey_id`** (int): ID do survey no Golden Truth
- **`psg_node`** (MarkdownNode): Nó raiz da árvore de markdown do artigo gerado

### Retorno

Retorna um dicionário com:
```python
{
    "gt_titles": List[str],           # Títulos do artigo GT
    "llm_titles": List[str],          # Títulos do artigo LLM
    "comparisons": List[dict]         # Lista de comparações ordenadas por distância
}
```

Cada comparação contém:
```python
{
    "gt_title": str,        # Título do GT
    "llm_title": str,       # Título correspondente da LLM
    "distance": float,      # Distância coseno (0 = idêntico, 2 = completamente diferente)
    "similarity": float     # Similaridade coseno (-1 a 1)
}
```

## Como Usar

### Método 1: Integrado com `single_eval()`

Adicione `"Compare_Section_Titles"` à lista de avaliações:

```python
evaluator = SurGEvaluator(
    device="0",
    survey_path="data/surveysMatScience.json",
    corpus_path="data/corpusMatScience.json",
    using_openai=True,
    api_key="your_api_key"
)

result = evaluator.single_eval(
    survey_id=26,
    passage_path="./caminho/para/arquivo.md",
    eval_list=["Compare_Section_Titles"]
)
```

### Método 2: Chamar Diretamente

```python
psg_node = markdownParser.parse_markdown("./caminho/para/arquivo.md")
result = evaluator.compare_section_titles(survey_id=26, psg_node=psg_node)
```

### Método 3: Com Todas as Avaliações

```python
result = evaluator.eval_all(
    passage_dir="./baselines/ID/output",
    eval_list=["ALL"],  # Inclui Compare_Section_Titles
    save_path="./output/log.json"
)
```

## Exemplo de Output

```
================================================================================
COMPARAÇÃO DE TÍTULOS DE SEÇÕES - Survey ID: 26
================================================================================
Títulos do Golden Truth (GT): 3
  1. Introduction
  2. Methods
  3. Conclusion

Títulos do Artigo LLM: 4
  1. Introduction to Deep Learning
  2. Methodology and Approaches
  3. Results and Analysis
  4. Conclusion and Future Work

--------------------------------------------------------------------------------
RESULTADOS DA COMPARAÇÃO (Ordenados por menor distância):
--------------------------------------------------------------------------------
Distância    Similaridade   Título GT                            Título LLM
--------------------------------------------------------------------------------
0.0312       0.9688         Introduction                         Introduction to Deep Learning
0.1456       0.8544         Methods                              Methodology and Approaches
0.2847       0.7153         Conclusion                           Conclusion and Future Work
================================================================================
```

## Dependências Necessárias

As seguintes bibliotecas devem estar instaladas:
- `numpy` - para operações com vetores
- `scipy` - para cálculos de distância (se necessário)
- `FlagEmbedding` - para gerar embeddings
- `sentence_transformers` - dependência do FlagModel

Todas já estão listadas no `requirements.txt`.

## Importações Adicionadas

No início do arquivo `evaluator.py` foram adicionadas:

```python
import numpy as np
from scipy.spatial.distance import cdist
```

## Características Técnicas

1. **Normalização de Embeddings**: Os embeddings são normalizados para garantir que o cálculo de similaridade coseno seja preciso
   
2. **Filtragem de Seções**: Apenas seções com mais de 100 caracteres são consideradas no GT para evitar ruído

3. **Correspondência One-to-One**: Cada título do GT é correspondido apenas ao título LLM mais similar

4. **Ordenação automática**: Resultados são exibidos em ordem crescente de distância

5. **Tratamento de Casos Especiais**: 
   - Filtro de títulos "root" e "Abstract:"
   - Validação de entrada antes do processamento
   - Aviso quando não há títulos suficientes

## Métricas de Distância

- **Distância = 0.0**: Títulos praticamente idênticos
- **Distância = 0.5**: Títulos moderadamente semelhantes
- **Distância = 1.0**: Títulos completamente diferentes (ortogonais)
- **Distância > 1.0**: Títulos com sentidos opostos

## Notas

- A função é chamada automaticamente quando `"ALL"` é incluído em `eval_list`
- O FlagModel é carregado apenas uma vez e reutilizado para melhor performance
- Os embeddings são computados em tempo de execução (sem cache persistente)
