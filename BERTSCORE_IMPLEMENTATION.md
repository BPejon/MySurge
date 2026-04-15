# 📋 IMPLEMENTAÇÃO: Análise de Seções com BERTScore

## ✅ Status: CONCLUÍDO COM SUCESSO

---

## 📦 Funções Implementadas

### 1. **`extract_section_text_from_markdown()`** 
📄 Arquivo: [src/structureFuncs.py](src/structureFuncs.py#L285-L320)

Extrai o texto de conteúdo de uma seção MarkdownNode e suas subsecções.

**Entrada:** MarkdownNode (seção LLM parseada)  
**Saída:** str (texto concatenado da seção)

```python
# Exemplo de uso
from structureFuncs import extract_section_text_from_markdown
text = extract_section_text_from_markdown(markdown_node)
```

---

### 2. **`calculate_bertscore_for_sections()`**
📄 Arquivo: [src/rougeBleuFuncs.py](src/rougeBleuFuncs.py#L37-L75)

Calcula BERTScore entre dois textos de seções usando modelo roberta-large.

**Entrada:** 
- `text_llm` (str): Texto gerado pela LLM
- `text_gt` (str): Texto Ground Truth

**Saída:** dict com `precision`, `recall`, `f1`

```python
# Exemplo de uso
from rougeBleuFuncs import calculate_bertscore_for_sections
result = calculate_bertscore_for_sections(text_llm, text_gt)
print(f"F1-Score: {result['f1']:.4f}")
```

---

### 3. **`extract_section_text_from_gt()`**
📄 Arquivo: [src/evaluator.py](src/evaluator.py#L206-L225)

Extrai o texto de conteúdo de uma seção do Ground Truth (estrutura JSON).

**Entrada:** dict (seção com chave 'content')  
**Saída:** str ou NaN

---

### 4. **`generate_comparison_table()`**
📄 Arquivo: [src/evaluator.py](src/evaluator.py#L267-L330)

Gera tabela comparativa entre seções GT e LLM com BERTScore.

**Entrada:**
- `comparison_result` (dict): Resultado de `compare_section_titles()`
- `psg_node` (MarkdownNode): Raiz do artigo LLM

**Saída:** pandas.DataFrame com 7 colunas:
1. `seção_llm` - Título da seção LLM
2. `seção_gt` - Título da seção GT
3. `distância` - Distância coseno entre títulos
4. `similaridade` - Similaridade coseno entre títulos
5. `bertscore_f1` - F1-Score do BERTScore
6. `texto_llm` - Preview do texto LLM
7. `texto_gt` - Preview do texto GT

```python
# Exemplo de uso
df = evaluator.generate_comparison_table(comparison_result, psg_node)
print(df)
```

---

### 5. **`print_comparison_table()`**
📄 Arquivo: [src/evaluator.py](src/evaluator.py#L332-L360)

Imprime a tabela comparativa em formato legível e exibe estatísticas.

**Saída:** Tabela formatada com 200 caracteres de largura

```python
# Exemplo de uso
evaluator.print_comparison_table(df)
```

---

## 🔧 Funções de Suporte

### `find_section_by_title(title, section_list)`
Busca uma seção pelo título na lista de seções GT.

### `find_markdown_section_by_title(title, markdown_node)`
Busca um nó MarkdownNode pelo título na árvore de markdown (BFS).

---

## 🔗 Integração no Workflow

**Localização:** [src/evaluator.py](src/evaluator.py#L376-L388) - método `single_eval()`

Quando `"Compare_Section_Titles"` está em `eval_list`:

```python
if "Compare_Section_Titles" in eval_list or "ALL" in eval_list:
    title_comparison_result = self.compare_section_titles(survey_id, psg_node)
    
    # Novo: Análise com BERTScore
    try:
        comparison_df = self.generate_comparison_table(title_comparison_result, psg_node)
        if not comparison_df.empty:
            self.print_comparison_table(comparison_df)
    except Exception as e:
        print(f"Erro ao gerar tabela de comparação com BERTScore: {e}")
```

---

## 📊 Exemplo de Saída

```
================================================================================
ANÁLISE COMPARATIVA: BERTSCORE E SIMILARIDADE DE SEÇÕES
================================================================================
Seção LLM                            Seção GT                             Dist     Sim      F1       Texto LLM                                           Texto GT
...
---
Total de seções comparadas: 15
BERTScore F1 Médio: 0.8342
Similaridade Média: 0.9127
```

---

## 🧪 Testes e Validação

✅ **Testes Implementados:**
- [x] Extração de texto de MarkdownNode
- [x] Cálculo de BERTScore (F1: 0.9173 em teste)
- [x] Comparação de títulos de seções
- [x] Geração de tabela comparativa
- [x] Impressão formatada de resultados

**Executar testes:**
```bash
python3 test_bertscore_integration.py
```

---

## 🚀 Como Usar

### Opção 1: Via `single_eval()`

```python
from evaluator import SurGEvaluator
import markdownParser

evaluator = SurGEvaluator(surveys, corpus, api_key)

# Executar avaliação com Compare_Section_Titles
result = evaluator.single_eval(
    survey_id=26,
    passage_path="path/to/markdown.md",
    eval_list=["Compare_Section_Titles", "SH-Recall"]
)
```

### Opção 2: Chamar diretamente

```python
from evaluator import SurGEvaluator
from markdownParser import parse_markdown

evaluator = SurGEvaluator(surveys, corpus, api_key)
psg_node = parse_markdown("path/to/markdown.md")

# 1. Comparar títulos
comparison_result = evaluator.compare_section_titles(survey_id, psg_node)

# 2. Gerar tabela com BERTScore
comparison_df = evaluator.generate_comparison_table(comparison_result, psg_node)

# 3. Imprimir resultados
evaluator.print_comparison_table(comparison_df)
```

---

## 📝 Notas Importantes

1. **BERTScore:** Usa modelo `roberta-large` por padrão. Automáticamente detecta GPU se disponível.

2. **Texto Completo:** Por decisão do usuário, o texto completo das seções é mantido (sem truncagem), com valores NaN para seções sem texto.

3. **Matching 1:1:** Cada título GT é pareado com único texto LLM mais similar (reutilizando similaridade já calculada).

4. **Saída:** Apenas print() na tela (sem exportação para arquivo por padrão).

5. **Performance:** 
   - Tempo de BERTScore ≈ 1-2 segundos por seção
   - Extração de texto: < 100ms

---

## 📚 Dependências

```
bert-score (já instalado)
torch
transformers
pandas
numpy
```

---

## ✨ Próximos Passos (Sugestões)

1. Adicionar opção de exportar tabela em CSV/JSON
2. Adicionar filtro por threshold de BERTScore F1
3. Gerar gráficos comparativos (seaborn/matplotlib)
4. Adicionar suporte para comparação entre múltiplos artigos LLM
5. Cache de embeddings para acelerar processamento

---

## 📦 Arquivos Modificados

- ✅ [src/structureFuncs.py](src/structureFuncs.py) - `extract_section_text_from_markdown()`
- ✅ [src/rougeBleuFuncs.py](src/rougeBleuFuncs.py) - `calculate_bertscore_for_sections()`
- ✅ [src/evaluator.py](src/evaluator.py) - Funções adicionadas e integração
- ✅ [test_bertscore_integration.py](test_bertscore_integration.py) - Script de testes

---

**Status:** ✅ Pronto para uso!
