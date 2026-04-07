# Quick Start: Comparação de Títulos de Seções

## Inicialização Rápida

### 1. Método Mais Simples: Usar a CLI

```bash
cd /home/breno/Documentos/onealSurge
python3 -c "
import sys
sys.path.insert(0, 'src')
from evaluator import SurGEvaluator
import markdownParser

# Inicializa
evaluator = SurGEvaluator(
    device='0',
    survey_path='data/surveysMatScience.json',
    corpus_path='data/corpusMatScience.json',
    using_openai=False
)

# Testa
psg_node = markdownParser.parse_markdown('./baselines/ID/output/26/0.md')
result = evaluator.compare_section_titles(26, psg_node)
"
```

### 2. Executar Script de Teste

```bash
cd /home/breno/Documentos/onealSurge
python3 test_section_titles.py
```

### 3. Executar Exemplos

```bash
cd /home/breno/Documentos/onealSurge
python3 examples_section_titles.py
```

### 4. Usar diretamente no seu código

```python
from src.evaluator import SurGEvaluator

# Inicializa o avaliador
evaluator = SurGEvaluator(
    device="0",
    survey_path="data/surveysMatScience.json",
    corpus_path="data/corpusMatScience.json",
    using_openai=False
)

# Opção A: Chamar diretamente
from src import markdownParser
psg_node = markdownParser.parse_markdown("caminho/para/arquivo.md")
result = evaluator.compare_section_titles(survey_id=26, psg_node=psg_node)

# Opção B: Via single_eval
result = evaluator.single_eval(
    survey_id=26,
    passage_path="caminho/para/arquivo.md",
    eval_list=["Compare_Section_Titles"]
)

# Opção C: Via eval_all
evaluator.eval_all(
    passage_dir="./baselines/ID/output",
    eval_list=["Compare_Section_Titles"],
    save_path="./output/log.json"
)
```

## O que a função faz

1. **Extrai títulos do Golden Truth** do arquivo JSON de surveys
2. **Extrai títulos do artigo LLM** do arquivo markdown gerado
3. **Gera representações vetoriais** (embeddings) de cada título
4. **Calcula similaridade coseno** entre todos os pares
5. **Encontra a melhor correspondência** (título LLM mais próximo para cada título GT)
6. **Exibe uma tabela** com os resultados ordenados por distância

## Interpretar os Resultados

```
Distância    | Significado
-------------|----------------------------------
0.0 - 0.15   | ✓ Excelente correspondência
0.15 - 0.30  | ✓ Boa correspondência
0.30 - 0.50  | ○ Correspondência moderada
0.50+        | ✗ Má correspondência
```

**Dica**: Uma distância baixa significa que os embeddings dos títulos são muito similares, indicando que o título gerado preservou bem o significado do original.

## Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `test_section_titles.py` | Script de teste básico |
| `examples_section_titles.py` | 4 exemplos práticos |
| `SECTION_TITLES_COMPARISON_DOC.md` | Documentação completa |
| `MUDANCAS_IMPLEMENTADAS.md` | Resumo técnico das mudanças |
| `QUICK_START.md` | Este arquivo |

## Estrutura de Retorno

```python
{
    "gt_titles": [
        "Introduction",
        "Methods", 
        "Results"
    ],
    "llm_titles": [
        "Intro to Topic",
        "Methodology",
        "Findings",
        "Discussion"
    ],
    "comparisons": [
        {
            "gt_title": "Introduction",
            "llm_title": "Intro to Topic",
            "distance": 0.0847,
            "similarity": 0.9153
        },
        {
            "gt_title": "Methods",
            "llm_title": "Methodology",
            "distance": 0.1234,
            "similarity": 0.8766
        },
        # ... mais comparações
    ]
}
```

## Usando em seu Script Existente

Se você já tem um script que usa `SurGEvaluator`, basta adicionar `"Compare_Section_Titles"` ao `eval_list`:

**Antes:**
```python
result = evaluator.single_eval(
    survey_id=26,
    passage_path="path/to/file.md",
    eval_list=["SH-Recall", "ROUGE-BLEU"]
)
```

**Depois:**
```python
result = evaluator.single_eval(
    survey_id=26,
    passage_path="path/to/file.md",
    eval_list=["SH-Recall", "ROUGE-BLEU", "Compare_Section_Titles"]
)
```

## Troubleshooting

### Erro: "Arquivo não encontrado"
- Verifique se o caminho está correto
- Use caminhos absolutos se possível

### Erro: "memory error" ao carregar modelo
- Reduza o uso de VRAM diminuindo o `device`
- Ou defina `device="-1"` para usar CPU

### Embeddings muito lentos
- Primeira execução é mais lenta (carrega o modelo)
- Execuções subsequentes reutilizam o modelo carregado

### Resultados muito diferentes do esperado
- Verifique se os títulos GT estão sendo extraídos corretamente
- Confirme o formato do arquivo markdown

## Contato

Para dúvidas adicionais, consulte:
- `SECTION_TITLES_COMPARISON_DOC.md` - Documentação técnica completa
- `examples_section_titles.py` - Exemplos de código
