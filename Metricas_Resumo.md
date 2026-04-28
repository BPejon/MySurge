# Métricas de Avaliação

## Visão Geral

O sistema avalia artigos gerados por LLM (AGLLM) comparando com Golden Truth (GT) usando **3 categorias** com **13 métricas** totais.

---

## Information Collection (5 métricas)

### Coverage

* Fração de Artigos citados no AGLLM que também são citados no GT.
* Essa métrica avalia se o AGLLM referenciou os artigos que o GT referenciou
* **Fórmula**: `artigos com o mesmo nome no GT e AGLLM/ total de artigos no GT`
* **Intervalo**: 0-1
* **Interpretação**: >0.85 = excelente | 0.50-0.70 = moderado | <0.30 = fraco

### Paper Level Relevance

* Avalia se os artigos citados pelo LLM são relevantes para o AGLLM como um todo, comparando com o GT. Usa **Natural Language Inference (NLI)** para determinar se um artigo, baseado em seu título e abstract, é semanticamente relevante para ser citado em um survey sobre o tópico.
* **Método**: Modelo CrossEncoder `cross-encoder/nli-deberta-v3-base`
* **Intervalo**: 0-1
* **Nota**: Se Entailment > Contradiction = relevante

### Section Level Relevance (Colocado no Surge, porém não estamos utilizando para relevancia)

* Relevância das citações para seções específicas
* **Intervalo**: 0-1
* **Avalia**: Coesão interna do survey

### Sentence Level Relevance (Colocado no Surge, porém não estamos utilizando para relevancia)

* Relevância das citações para sentença específica (mais rigoroso)
* **Intervalo**: 0-1
* **Uso**: Qualidade precisão de citação

---

## Survey Structure

### SH-Recall (Soft Heading Recall)

* Fração de títulos GT recuperados semanticamente
* **Encoder**: `BAAI/bge-large-en-v1.5`
* **Fórmula**: Média das máximas similaridades coseno por título
* **Intervalo**: 0-1

### Structure Quality

* Avaliação do LLM as a Judge da hierarquia do survey
* **Juiz**: GPT-4 (OpenAI API)
* **Intervalo**: 0-5 (escala ordinal)
* **Critérios**: Similaridade estrutural + correspondência de tópicos

### Subtitle Similarity

* Proporção de títulos LLM com correspondência no GT
* **Threshold**: 0.85 (similaridade de coseno)
* **Fórmula**: `correspondência acima do limiar/total de títulos no AGLLM`
* **Intervalo**: 0-1

---

## Survey Content

### ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)

* Overlap de n-gramas entre textos
* **Lib**: `rouge_score` (com stemming)
* **Diferença**:
  * ROUGE-1: palavras individuais
  * ROUGE-2: pares de palavras
  * ROUGE-L: sequência comum mais longa
* **Intervalo**: 0-1

### BLEU

* Precisão de n-gramas do LLM no GT
* **Lib**: `sacrebleu`
* **Intervalo**: 0-100
* **Penalidade**: Brevity penalty para textos muito curtos

### Content LLM as Judge

* Avaliação LLM as a Judge de cobertura e profundidade
* **Juiz**: GPT-4
* **Critérios**:
  * Comprehensiveness (35 pts)
  * Discussion Depth (35 pts)
  * Content Balance (30 pts)
* **Intervalo**: 0-100

### BERTScore

* Similaridade semântica via embeddings BERT dos textos com títulos semelhantes
* **Modelo**: `roberta-large`
* **Retorna**: Precision, Recall, F1 (0-1)

## Como Interpretar

* **Valores altos** em Information_Collection: LLM cita artigos relevantes do GT
* **Valores altos** em Survey_Structure: Estrutura LLM similar ao GT
* **Valores altos** em Survey_Content: Conteúdo textual similar ao GT
* **Score geral**: Média ponderada das 3 categorias

---

**Versão**: 1.0

**Data**: 23/04/2026
