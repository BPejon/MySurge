# Métricas de Avaliação - Versão Resumida

## Visão Geral

O sistema avalia surveys gerados por LLM comparando com Golden Truth (GT) usando **3 categorias** com **13 métricas** totais.

---

## Information Collection (5 métricas)

### Coverage
- **O que é**: Fração de artigos do GT citados no LLM
- **Fórmula**: `hit / total_gt_articles`
- **Intervalo**: 0-1 (0.60-0.90 esperado)
- **Interpretação**: >0.85 = excelente | 0.50-0.70 = moderado | <0.30 = fraco

### Paper Level Relevance
- **O que é**: Relevância dos artigos citados para o survey (NLI-based)
- **Método**: Modelo CrossEncoder `cross-encoder/nli-deberta-v3-base`
- **Intervalo**: 0-1 (0.70-0.95 esperado)
- **Nota**: Entailment > Contradiction = relevante

### Section Level Relevance
- **O que é**: Relevância das citações para seções específicas
- **Intervalo**: 0-1 (0.60-0.90 esperado)
- **Avalia**: Coesão interna do survey

### Sentence Level Relevance
- **O que é**: Relevância das citações para sentença específica (mais rigoroso)
- **Intervalo**: 0-1 (0.50-0.85 esperado)
- **Uso**: Qualidade precisão de citação

---

## Survey Structure

### SH-Recall (Soft Heading Recall)
- **O que é**: Fração de títulos GT recuperados semanticamente
- **Encoder**: `BAAI/bge-large-en-v1.5`
- **Fórmula**: Média das máximas similaridades coseno por título
- **Intervalo**: 0-1 (0.75-0.95 esperado)
- **Score**: 0.95+ = excelente | 0.60-0.79 = bom | <0.40 = fraco

### Structure Quality
- **O que é**: Avaliação holística da hierarquia do survey
- **Juiz**: GPT-4 (OpenAI API)
- **Intervalo**: 0-5 (escala ordinal)
- **Critérios**: Similaridade estrutural + correspondência de tópicos
- **Score**: 4-5 = excelente | 2.5-3.4 = bom | 0-1.4 = fraco

### Subtitle Similarity
- **O que é**: Proporção de títulos LLM com correspondência no GT
- **Threshold**: 0.85 (similarity coseno)
- **Fórmula**: `matches_above_threshold / total_llm_titles`
- **Intervalo**: 0-1 (0.40-0.80 esperado)

---

## Survey Content

### ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- **O que é**: Overlap de n-gramas entre textos
- **Lib**: `rouge_score` (com stemming)
- **Diferença**:
  - ROUGE-1: palavras individuais (recall-based)
  - ROUGE-2: pares de palavras (fluidez)
  - ROUGE-L: sequência comum mais longa (ordem)
- **Intervalo**: 0-1 (0.60-0.85 esperado)

### BLEU
- **O que é**: Precisão de n-gramas do LLM no GT
- **Lib**: `sacrebleu` (precision-based, oposto ao ROUGE)
- **Intervalo**: 0-100 (20-60 esperado)
- **Penalidade**: Brevity penalty para textos muito curtos

### Content LLM as Judge
- **O que é**: Avaliação holística de cobertura/profundidade
- **Juiz**: GPT-4
- **Critérios**:
  - Comprehensiveness (35 pts)
  - Discussion Depth (35 pts)
  - Content Balance (30 pts)
- **Intervalo**: 0-100 (60-90 esperado)

### BERTScore
- **O que é**: Similaridade semântica via embeddings BERT
- **Modelo**: `roberta-large`
- **Retorna**: Precision, Recall, F1 (0-1)
- **Interpretação**: 0.90+ = excelente alinhamento

---

## Exemplo de Saída

```json
{
  "Information_Collection": {
    "Coverage": 0.75,
    "Relevance": {
      "Paper_Level": 0.82,
      "Section_Level": 0.78,
      "Sentence_Level": 0.71
    }
  },
  "Survey_Structure": {
    "SH-Recall": 0.86,
    "Structure_Quality": 4.0,
    "Subtitle_similarity": 0.68
  },
  "Survey_Content": {
    "ROUGE-1": 0.72,
    "ROUGE-2": 0.58,
    "ROUGE-L": 0.65,
    "BLEU": 42.3,
    "Content LLM as Judge": 78
  }
}
```

---

## Resumo de Valores Esperados

| Categoria | Métrica | Range |
|-----------|---------|-------|
| Info | Coverage | 0.60-0.90 |
| Info | Paper Relevance | 0.70-0.95 |
| Info | Section Relevance | 0.60-0.90 |
| Info | Sentence Relevance | 0.50-0.85 |
| Structure | SH-Recall | 0.75-0.95 |
| Structure | Structure Quality | 3-5 |
| Structure | Subtitle Similarity | 0.40-0.80 |
| Content | ROUGE-1 | 0.60-0.85 |
| Content | ROUGE-2 | 0.40-0.70 |
| Content | ROUGE-L | 0.50-0.75 |
| Content | BLEU | 25-60 |
| Content | Content Quality | 60-90 |

---

## Como Interpretar

- **Valores altos** em Information_Collection: LLM cita artigos relevantes do GT
- **Valores altos** em Survey_Structure: Estrutura LLM similar ao GT
- **Valores altos** em Survey_Content: Conteúdo textual similar ao GT
- **Score geral**: Média ponderada das 3 categorias

---

**Versão**: 1.0 Resumida | **Linhas**: ~185 | **Data**: 17/04/2026
