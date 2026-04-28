# Documentação de Métricas de Avaliação de Surveys

## 📋 Índice

1. [Introdução](#introdução)
2. [Information Collection - Coleta de Informação](#information-collection---coleta-de-informação)
3. [Survey Structure - Estrutura do Survey](#survey-structure---estrutura-do-survey)
4. [Survey Content - Conteúdo do Survey](#survey-content---conteúdo-do-survey)
5. [Exemplo de Saída JSON](#exemplo-de-saída-json)
6. [Conclusão](#conclusão)

---

## Introdução

Este documento descreve as **três categorias principais de métricas** utilizadas no sistema de avaliação de surveys gerados por LLM (Large Language Models) comparando com o **Golden Truth (GT)** - a versão de referência criada manualmente.

O objetivo dessas métricas é avaliar:

- **Qualidade de coleta de informações**: Se o survey LLM cita os mesmos artigos relevantes do GT
- **Qualidade estrutural**: Se a organização e os títulos das seções são similares ao GT
- **Qualidade de conteúdo**: Se o conteúdo textual é semanticamente similar e relevante

Os valores metricamente variam entre **0 e 1** (ou 0 a 100 em alguns casos), onde valores mais altos indicam melhor alinhamento com o Golden Truth.

---

## Information Collection - Coleta de Informação

### 📊 Introdução à Categoria

A categoria **Information_Collection** avalia como o artigo gerado pelas LLM coleta e referencia informações do corpus de documentos, comparando com o Golden Truth. Existem duas subcategorias principais:

1. **Comprehensiveness** (Compreensibilidade): O quão bem o LLM recupera as referências do GT
2. **Relevance** (Relevância): O quão relevantes são as referências citadas em diversos níveis

---

### 1. Coverage (Cobertura)

#### O que é?

**Coverage** mede a **fração de artigos citados no Golden Truth que foram também citados no artigo gerado pelas LLM**. É uma métrica que responde: "O survey gerado referenciou os artigos principais que o GT referenciou?"

#### Definição Formal

$$\text{Coverage} = \frac{\text{Número de citações do GT encontradas no LLM}}{\text{Número total de citações no GT}}$$

#### Como é Calculada?

O algoritmo de cálculo é simples:

1. Extrai a lista de **todos os artigos citados no GT** (`target_cites`)
2. Extrai a lista de **todos os artigos citados no LLM** (`gen_cite_map`)
3. Para cada artigo no GT, verifica se foi citado no LLM
4. Conta os matches e divide pelo total de artigos no GT

```python
# Pseudo-código da implementação
def eval_coverage(target_cites, gen_cite_map):
    all = len(target_cites)
    hit = 0
    for cite in target_cites:
        if cite in gen_cite_map.values():
            hit += 1
    return hit / all  # Retorna valor entre 0 e 1
```

#### Localização no Código

- **Arquivo**: `src/informationFuncs.py`
- **Função**: `eval_coverage(target_cites, gen_cite_map)`
- **Linhas**: 7-20

#### Interpretação dos Valores

| Valor | Interpretação |
| :--- | :--- |
| 0.90 - 1.00 | Excelente cobertura - LLM citou praticamente todos os artigos principais |
| 0.70 - 0.89 | Boa cobertura - LLM citou a maioria dos artigos importantes |
| 0.50 - 0.69 | Cobertura moderada - LLM perdeu alguns artigos importantes |
| 0.30 - 0.49 | Cobertura fraca - Muitos artigos importantes foram omitidos |
| 0.00 - 0.29 | Cobertura muito fraca - LLM citou muito poucos dos artigos do GT |

#### Exemplo

- **GT cita**: 20 artigos diferentes
- **LLM cita**: 18 artigos, dos quais 16 estão no GT
- **Coverage**: 16/20 = **0.80** (80%)

#### Relação com Comparação LLM vs GT

Essa métrica é fundamental para entender se o LLM consegue identificar quais são os **trabalhos principais da área**. Um Coverage alto significa que a LLM tem uma boa compreensão do domínio.

---

### 2. Paper_Level Relevance (Relevância em Nível de Artigo)

#### O que é?

**Paper_Level Relevance** avalia se **os artigos citados pelo LLM são relevantes para o survey como um todo**, comparando com o GT. Usa **Natural Language Inference (NLI)** para determinar se um artigo (por sua título e abstract) é semanticamente relevante para ser citado em um survey sobre o tópico.

#### Como é Calculada?

O cálculo utiliza um modelo **CrossEncoder** treinado em NLI (Natural Language Inference):

1. Para cada artigo citado pelo LLM:
   - **Se encontrado no corpus**: Extrai título e abstract do corpus
   - **Se não encontrado**: Marca como "[NOTEXIST]"
2. Para cada artigo, cria um par de sentenças:
   - **Sentença 1**: "There is a paper. Title: '{title}'. Abstract: '{abstract}'"
   - **Sentença 2**: "The paper titled '{title}' with the given abstract could be cited in the paper: '{survey_title}'."
3. O modelo NLI prevê scores para 3 classes:
   - **Contradiction** (C): Artigo não é relevante
   - **Entailment** (E): Artigo é relevante
   - **Neutral** (N): Relação neutra
4. Conta como relevante se:
   - E > C AND E > N (Entailment vence) → +1 ponto
   - N > C AND N > E (Neutral vence mas E > C) → +0.5 pontos
5. Divida pelos total de artigos citados

#### Localização no Código

- **Arquivo**: `src/informationFuncs.py`
- **Função**: `eval_relevance_paper(target_survey, gen_cite_map, cite_content, nli_model)`
- **Linhas**: 24-64
- **Modelo NLI**: `cross-encoder/nli-deberta-v3-base`

#### Interpretação dos Valores

| Valor | Significado |
| :--- | :--- |
| 0.90 - 1.00 | Praticamente todos os artigos citados são relevantes |
| 0.70 - 0.89 | A maioria dos artigos é relevante ao survey |
| 0.50 - 0.69 | Alguns artigos são relevantes, mas há ruído |
| 0.30 - 0.49 | Muitos artigos não são relevantes à temática |
| 0.00 - 0.29 | Maioria das citações não é relevante |

#### Exemplo

- **LLM cita**: 15 artigos
- **Validação**:
  - 10 artigos estão no corpus GT → todos relevantes = 10 pontos
  - 5 artigos não estão no corpus → 3 com NLI positivo = 3 pontos
- **Paper_Level Relevance**: (10 + 3) / 15 = **0.87** (87%)

#### Relação com Comparação LLM vs GT

Se a Relevância em Nível de Paper é alta, significa que o LLM não está citando artigos aleatoriamente, mas realmente selecionando aqueles que são relevantes ao tópico principal. Valores baixos indicam que o LLM pode estar alucinando referências ou citando trabalhos fora do escopo.

---

### 3. Section_Level Relevance (Relevância em Nível de Seção)

#### O que é?

**Section_Level Relevance** avalia se **as citações estão sendo usadas em seções apropriadas**. Mede se um artigo (definido por título + abstract) é relevante para a **seção específica (subsection)** onde foi citado.

#### Como é Calculada?

Processo similar ao Paper_Level, mas com contexto de seção:

1. Extrai todo citação com: `(referência_id, título_seção, sentença_contendo_citação)`
2. Para cada citação:
   - **Sentença 1**: Artigo com título e abstract
   - **Sentença 2**: "The paper titled '{title}' ... is relevant to the section: '{section_title}'."
3. Usa modelo NLI para avaliar relevância
4. Contas hits da mesma forma que Paper_Level
5. Divide pelo número total de citações extraídas

#### Localização no Código

- **Arquivo**: `src/informationFuncs.py`
- **Função**: `eval_relevance_section(nli_pairs_origin, nli_model)`
- **Linhas**: 70-100

#### Interpretação dos Valores

Mesma escala que Paper_Level Relevance (0-1), mas avaliando contexto mais específico.

#### Exemplo

- Uma seção sobre "Redes Neurais Convolucionais"
- Artigo citado: "Attention is All You Need" (sobre Transformers)
- **Section_Level Relevance**: Pode ser baixa se Transformers não são diretamente relevantes àquela seção

#### Relação com Comparação LLM vs GT

Section_Level Relevance mede a **coesão interna** do survey. Se for alta, significa que o LLM está citando artigos relevantes não apenas globalmente, mas no contexto específico de cada seção. Valores baixos indicam citações "soltas" ou deslocadas.

---

### 4. Sentence_Level Relevance (Relevância em Nível de Sentença)

#### O que é?

**Sentence_Level Relevance** é o nível mais granular. Avalia se um artigo citado é relevante **para a sentença específica** onde aparece a citação. É a forma mais rigorosa de avaliar qualidade de citação.

#### Como é Calculada?

Processo idêntico ao Section_Level, mas comparando:

- **Sentença 1**: Artigo (título + abstract)
- **Sentença 2**: "The paper titled '{title}' ... could be cited in the sentence: '{sentence}'."

Onde `sentence` é a sentença exata contendo a citação `[ref_id]`.

#### Localização no Código

- **Arquivo**: `src/informationFuncs.py`
- **Função**: `eval_relevance_sentence(nli_pairs_origin, nli_model)`
- **Linhas**: 103-135

#### Interpretação dos Valores

Similar às métricas anteriores, mas com avaliação mais rigorosa.

#### Exemplo de Cenários

```text
Sentença: "Redes neurais revolucionaram a visão computacional [23]."
Artigo 23: "Deep Learning: Methods and Applications in Computer Vision"
→ Relevância ALTA

Sentença: "O Python é uma linguagem popular [42]."
Artigo 42: "Deep Learning: Methods and Applications in Computer Vision"
→ Relevância BAIXA (contexto desmatched)
```

#### Relação com Comparação LLM vs GT

Sentence_Level Relevance é o **indicador mais preciso de qualidade de citação**. Se for alta, o LLM está escolhendo exatamente os artigos certos para apoiar cada afirmação específica. Valores baixos indicam citações genéricas ou misapplied.

---

## Survey Structure - Estrutura do Survey

### 📐 Introdução à Categoria

A categoria **Survey_Structure** avalia a **qualidade e fidelidade da organização hierárquica do survey**, incluindo:

- Similaridade dos títulos das seções
- Recuperação (recall) semântica dos títulos
- Qualidade geral da estrutura

---

### 1. SH-Recall (Soft Heading Recall)

#### O que é?

**SH-Recall (Soft Heading Recall)** mede a **fração de títulos de seções do Golden Truth que foram "recuperados" no artigo gerado**, considerando similaridade semântica (não apenas matches exatos).

É chamado "soft" porque usa embeddings semânticos ao invés de comparação exata de strings.

#### Definição Formal

$$\text{SH-Recall} = \frac{\sum_{i=1}^{|G|} \max_j \text{similarity}(G_i, P_j)}{|G|}$$

Onde:

- $G$ = conjunto de títulos do Golden Truth
- $P$ = conjunto de títulos gerados pela LLM
- $\text{similarity}(G_i, P_j)$ = similaridade semântica entre dois títulos
- $\max_j$ = máxima similaridade encontrada com qualquer título gerado

#### Como é Calculada?

1. **Gera embeddings** para todos os títulos usando modelo FlagEmbedding (`BAAI/bge-large-en-v1.5`)
2. **Calcula matriz de similaridade coseno**:
   - Matriz de tamanho `|G| x |P|`
   - Cada elemento é a similaridade entre um título GT e um título LLM
3. **Encontra máxima similaridade**:
   - Para cada título do GT, encontra o título LLM mais similar
   - Armazena essa máxima similaridade
4. **Calcula média**:
   - SH-Recall = média de todas as máximas similaridades

#### Localização no Código

- **Arquivo**: `src/structureFuncs.py`
- **Função**: `soft_heading_recall(G, P, model)`
- **Linhas**: 18-47
- **Modelo de embedding**: BAAI/bge-large-en-v1.5

#### Interpretação dos Valores

| Valor | Interpretação |
| :--- | :--- |
| 0.95 - 1.00 | Excelente - Praticamente todos os títulos foram recuperados semanticamente |
| 0.80 - 0.94 | Muito bom - Maioria dos títulos foram bem recuperados |
| 0.60 - 0.79 | Bom - Títulos foram recuperados mas com algumas variações |
| 0.40 - 0.59 | Razoável - Alguns títulos perdidos ou muito diferentes |
| 0.00 - 0.39 | Fraco - Muitos títulos não foram recuperados semanticamente |

#### Exemplo

**Títulos GT:**

1. "Introduction to Deep Learning"
2. "Convolutional Neural Networks"
3. "Applications in Computer Vision"

**Títulos LLM:**

1. "Overview of Deep Learning Fundamentals"
2. "CNN Architecture and Design"
3. "Computer Vision Applications and Implementations"
4. "Conclusion"

**Processo:**

- "Introduction to Deep Learning" vs melhor match "Overview of Deep Learning Fundamentals" → 0.92 similaridade
- "Convolutional Neural Networks" vs melhor match "CNN Architecture and Design" → 0.88 similaridade
- "Applications in Computer Vision" vs melhor match "Computer Vision Applications and..." → 0.85 similaridade

**SH-Recall = (0.92 + 0.88 + 0.85) / 3 = 0.88** (88%)

#### Relação com Comparação LLM vs GT

SH-Recall mede como a LLM **estrutura conceitualmente** o survey: se consegue identificar as principais seções e recuperá-las (mesmo que com nomes diferentes). Um valor alto indica que a LLM tem uma visão semelhante da estrutura do domínio.

---

### 2. Structure Quality (Qualidade da Estrutura)

#### O que é?

**Structure Quality** avalia a **semelhança geral da estrutura hierárquica** entre o artigo GT e o gerado, incluindo a relação entre títulos de seções, subseções e conteúdo. É uma avaliação holística feita por um modelo LLM (OpenAI GPT-4) como juiz.

#### Como é Calculada?

O processo utiliza um modelo LLM (GPT-4) como avaliador:

1. **Extrai estrutura do GT**:
   - Constrói uma representação hierárquica com indentação de Markdown
   - Exemplo: # Seção, ## Subsecção, ### Sub-subsecção
2. **Extrai estrutura do LLM**: Mesmo processo
3. **Cria prompt de comparação** com critérios de scoring (0-5)
4. **Envia para GPT-4**:
   - Modelo compara estruturas lado a lado
   - Avalia baseado em critérios de similaridade
5. **Retorna score**: Entre 0 e 5

#### Localização no Código

- **Arquivo**: `src/structureFuncs.py`
- **Função**: `eval_structure_quality_client(target_survey, psg_node, client)`
- **Linhas**: 250-267
- **Modelo**: OpenAI GPT-4 (via API OpenAI)

#### Critérios de Avaliação (do prompt)

| Score | Critério |
| :--- | :--- |
| 5 | Quase idêntico - Praticamente todos os títulos correspondem, estrutura idêntica |
| 4 | Muito similar - Maioria dos títulos equivalentes, poucas diferenças no rewording |
| 3 | Similar - Tópicos principais coincidentes, mas estrutura pode variar |
| 2 | Parcialmente similar - Alguns tópicos comuns, mas muitas diferenças estruturais |
| 1 | Pouco similar - Poucos tópicos em comum, estrutura muito diferente |
| 0 | Completamente diferente - Nenhuma relação estrutural |

#### Interpretação dos Valores

| Valor | Interpretação |
| :--- | :--- |
| 4.5 - 5.0 | Excelente - Estrutura praticamente idêntica |
| 3.5 - 4.4 | Muito bom - Estrutura bem similar com variações menores |
| 2.5 - 3.4 | Bom - Estrutura similar mas com algumas diferenças |
| 1.5 - 2.4 | Razoável - Estrutura parcialmente similiar |
| 0.0 - 1.4 | Fraco - Estrutura muito diferente |

#### Relação com Comparação LLM vs GT

Ao contrário de SH-Recall que mede recuperação, Structure Quality mede **organização geral e coesão** da estrutura do survey como um todo. Um score alto indica que a LLM compreendeu não apenas os tópicos, mas como organizá-los logicamente.

---

### 3. Subtitle Similarity (Similaridade de Subtítulos)

#### O que é?

**Subtitle Similarity** (também chamada de `subtitle_similarity`) mede a **fração de títulos gerados que têm correspondência com títulos do GT** usando threshold de similaridade semântica.

É uma métrica que combina elementos tanto de SH-Recall quanto de Structure Quality: verifica quantos títulos gerados têm um match "suficientemente bom" no GT.

#### Definição Formal

$$\text{Subtitle\_Similarity} = \frac{\text{Número de títulos LLM com } \text{similarity} > \text{threshold}}{\text{Número total de títulos LLM}}$$

Onde threshold padrão = 0.85

#### Como é Calculada?

1. **Compara títulos GT vs LLM** usando embeddings FlagEmbedding
2. **Calcula matriz de similaridade** coseno
3. **Para cada título GT**:
   - Encontra melhor match no LLM
   - Calcula distância: 1 - similaridade
4. **Filtra por threshold** (geralmente 0.85 de similaridade)
5. **Calcula proporção**:
   - Divide número de matches acima do threshold pelo total de títulos LLM

#### Localização no Código

- **Arquivo**: `src/evaluator.py`
- **Função**: `compare_section_titles(survey_id, psg_node)` (retorna dict com `subtitle_similarity`)
- **Linhas**: 92-160
- **Threshold usado**: 0.85

#### Interpretação dos Valores

| Valor | Interpretação |
| :--- | :--- |
| 0.90 - 1.00 | Excelente - Praticamente todos os títulos LLM têm correspondência |
| 0.70 - 0.89 | Boa - Maioria dos títulos têm correspondência |
| 0.50 - 0.69 | Moderada - Cerca de metade dos títulos correspondem |
| 0.30 - 0.49 | Fraca - Poucos títulos correspondem |
| 0.00 - 0.29 | Muito fraca - Muito poucos ou nenhum título correspondente |

#### Exemplo

**GT tem**: 5 títulos principais
**LLM gera**: 7 títulos

**Comparações** (com threshold 0.85):

- Título 1 LLM: 0.88 ✓
- Título 2 LLM: 0.82 ✗
- Título 3 LLM: 0.91 ✓
- Título 4 LLM: 0.79 ✗
- Título 5 LLM: 0.87 ✓
- Título 6 LLM: 0.93 ✓
- Título 7 LLM: 0.76 ✗

**Subtitle Similarity = 4 / 7 ≈ 0.57** (57%)

#### Relação com Comparação LLM vs GT

Subtitle Similarity é uma métrica **pragmática** que responde: "Quantos títulos gerados podem ser mapeados para o GT?" Um valor alto significa que a LLM está gerando uma estrutura que **pode ser facilmente alinhada** com o GT.

---

## Survey Content - Conteúdo do Survey

### 📝 Introdução à Categoria

A categoria **Survey_Content** avalia a **qualidade semântica e textual** do conteúdo gerado, incluindo:

- Sobreposição de n-gramas (ROUGE, BLEU)
- Similaridade baseada em embeddings (BERTScore)
- Avaliação de cobertura temática por LLM

---

### 1. ROUGE-1, ROUGE-2, ROUGE-L

#### O que é ROUGE?

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) é um conjunto de métricas usadas para avaliar qualidade de sumários e comparar conteúdo textual com referências. ROUGE mede **sobreposição de n-gramas** entre textos.

#### ROUGE-1 (Unigrams)

**O que é?**
ROUGE-1 mede a **sobreposição de palavras individuais (unigrams)** entre o texto gerado e o texto de referência.

**Definição Formal:**

$$\text{ROUGE-1} = \frac{\sum_{s \in S} \sum_{w \in s} \min(\text{count}_{\text{gen}}(w), \text{count}_{\text{ref}}(w))}{\sum_{s \in S} \sum_{w \in s} \text{count}_{\text{ref}}(w)}$$

Onde:

- Gen = texto gerado
- Ref = texto de referência
- $\text{count}(w)$ = número de vezes que palavra aparece

#### ROUGE-2 (Bigrams)

**O que é?**
ROUGE-2 mede a **sobreposição de pares de palavras consecutivas (bigrams)**.

Segue mesma fórmula que ROUGE-1 mas com bigrams ao invés de unigrams.

#### ROUGE-L (Longest Common Subsequence)

**O que é?**
ROUGE-L mede a **sequência comum mais longa (LCS)** entre textos. Captura similaridade considerando **ordem das palavras**.

Diferentemente de ROUGE-1 e ROUGE-2, ROUGE-L não requer palavras consecutivas, apenas que estejam na mesma ordem.

**Exemplo:**

```text
Referência: "The quick brown fox jumps over the lazy dog"
Gerado: "The brown fox jumps over dog"
LCS: "The brown fox jumps over dog" (7 palavras)
ROUGE-L: 7 / 9 ≈ 0.78
```

#### Como são Calculadas?

```python
def calculate_average_rouge_bleu(A, B):
    """
    A: Lista de textos gerados (seções do LLM)
    B: Lista de textos referência (seções do GT)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for text_gen in A:
        max_rouge1, max_rouge2, max_rougeL = 0, 0, 0

        # Encontra melhor match em B
        for text_ref in B:
            scores = scorer.score(text_gen, text_ref)
            max_rouge1 = max(max_rouge1, scores['rouge1'].fmeasure)
            max_rouge2 = max(max_rouge2, scores['rouge2'].fmeasure)
            max_rougeL = max(max_rougeL, scores['rougeL'].fmeasure)

        rouge1_scores.append(max_rouge1)
        rouge2_scores.append(max_rouge2)
        rougeL_scores.append(max_rougeL)

    # Retorna médias
    return mean(rouge1_scores), mean(rouge2_scores), mean(rougeL_scores)
```

#### Localização no Código

- **Arquivo**: `src/rougeBleuFuncs.py`
- **Função**: `calculate_average_rouge_bleu(A, B)`
- **Linhas**: 30-69
- **Biblioteca**: `rouge_score` (uso de stemming ativado)

#### Interpretação dos Valores

Para todas as métricas ROUGE (0 a 1):

| Valor | Interpretação |
| :--- | :--- |
| 0.80 - 1.00 | Excelente - Conteúdo muito similar ao GT |
| 0.60 - 0.79 | Bom - Sobreposição significativa |
| 0.40 - 0.59 | Medio - Alguma sobreposição |
| 0.20 - 0.39 | Fraco - Pouca sobreposição |
| 0.00 - 0.19 | Muito fraco - Quase nenhuma sobreposição |

#### Comparação entre as 3 ROUGE Metrics

| Métrica | Avalia | É Sensível a |
| :--- | :--- | :--- |
| ROUGE-1 | Palavras individuais | Vocabulário comum |
| ROUGE-2 | Pares de palavras | Fluidez e sequência |
| ROUGE-L | Sequência mais longa | Ordem e estrutura |

#### Exemplo Real

```text
Texto GT: "Deep learning models have revolutionized computer vision. CNNs are widely used."
Texto LLM: "Deep learning has changed computer vision. Convolutional neural networks are common."

ROUGE-1: Ambos têm "deep", "learning", "computer", "vision" → score alto (~0.75)
ROUGE-2: Menos bigrams em comum → score médio (~0.55)
ROUGE-L: Sequência similar mas com mudanças → score médio (~0.60)
```

#### Relação com Comparação LLM vs GT

ROUGE mede **similaridade lexical e sintática**. Valores altos indicam que o LLM consegue expressar ideias semelhantes usando vocabulário similar ao GT. Valores baixos podem indicar parafrasagem, omissão de conceitos ou completamente diferente.

---

### 2. BLEU Score

#### O que é?

**BLEU** (Bilingual Evaluation Understudy) é uma métrica de precisão que mede a **fração de palavras/frases do texto gerado que aparecem no texto de reférencia**. É comumente usada em machine translation.

Diferente de ROUGE (que é baseada em recall), BLEU é baseada em **precisão**.

#### Definição Formal

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Onde:

- BP = brevity penalty (penaliza textos muito curtos)
- $p_n$ = precisão do n-grama
- $w_n$ = peso do n-grama (tipicamente 0.25 para cada de 1-4)

#### Como é Calculada?

```python
def calculate_average_rouge_bleu(A, B):
    bleu_scores = []

    for text_gen in A:
        # Calcula BLEU comparando com TODOS os textos em B
        # e retorna a pontuação
        bleu_score = sacrebleu.sentence_bleu(text_gen, B).score
        bleu_scores.append(bleu_score)

    avg_BLEU = mean(bleu_scores)
    return avg_BLEU
```

#### Localização no Código

- **Arquivo**: `src/rougeBleuFuncs.py`
- **Função**: `calculate_average_rouge_bleu(A, B)` (linha de BLEU)
- **Linhas**: 30-69
- **Biblioteca**: `sacrebleu`

#### Interpretação dos Valores

| Valor (%) | Interpretação |
| :--- | :--- |
| 50 - 100 | Excelente - Conteúdo muito próximo ao GT |
| 35 - 49 | Bom - Sobreposição boa |
| 20 - 34 | Médio - Alguma sobreposição |
| 10 - 19 | Fraco - Pouca sobreposição |
| 0 - 9 | Muito fraco - Praticamente não há overlap |

#### Diferenças ROUGE vs BLEU

| Aspecto | ROUGE | BLEU |
| :--- | :--- | :--- |
| Tipo | Recall-based | Precision-based |
| Foco | O que GT tem, LLM recuperou? | O que LLM tem, está correto no GT? |
| Uso ideal | Resumos, compressão | Tradução, paráfrase |
| Penalidade | Por omissão | Por adição |

#### Exemplo

```text
Ref: "The cat sat on the mat" (5 palavras)
Gen: "The cat sat on the mat and slept" (8 palavras)

BLEU forte em overlap mas penaliza comprimento
ROUGE mais tolerante com comprimento diferente
```

#### Relação com Comparação LLM vs GT

BLEU responde: "Qual fração do conteúdo gerado é realmente necessária/está certa?" Um BLEU Alto pode significar que o LLM é conciso e preciso. Um BLEU baixo mas ROUGE alto pode indicar parafrasagem criativa.

---

### 3. Content LLM as a Judge

#### O que é?

**Content LLM as a Judge** (também chamada de **Structure Quality**) é uma avaliação **holística do conteúdo** feita por um modelo LLM (GPT-4) como juiz. Avalia a **qualidade geral de cobertura e profundidade** do conteúdo gerado.

#### Como é Calculada?

1. **Extrai todo conteúdo** das seções do survey LLM
2. **Concatena seções** em um documento único
3. **Limita tamanho** para 5000 caracteres (se necessário)
4. **Cria prompt de avaliação** com critérios específicos (0-100)
5. **Envia para GPT-4**: Avalia cobertura temática, profundidade, balanceamento
6. **Retorna score**: Entre 0 e 100

```python
def eval_content_client(psg_node, client):
    # Extrai conteúdo de todas as seções
    psgs = get_content_list(psg_node)
    full_article = "\n\n".join(psgs)

    # Limita a 5000 caracteres
    if len(full_article) > 5000:
        full_article = full_article[:5000]
        full_article = full_article[:full_article.rfind('.')]

    # Cria prompt de avaliação
    prompt = get_content_check_prompt(full_article)

    # Chama GPT-4
    score = chat_openai(prompt, client, 0)
    return score  # 0-100
```

#### Critérios de Avaliação

Os critérios avaliados (100 pontos total):

1. **Topic Comprehensiveness (35 pontos)**
   - Range de tópicos essenciais cobertos
   - Inclusão de áreas emergentes
   - Identificação de conceitos-chave
2. **Discussion Depth (35 pontos)**
   - Nível de detalhe na análise
   - Desenvolvimento de argumentos-chave
   - Profundidade de explicações
3. **Content Balance (30 pontos)**
   - Cobertura proporcional de tópicos
   - Distribuição apropriada de ênfase
   - Alocação lógica de espaço

#### Localização no Código

- **Arquivo**: `src/informationFuncs.py`
- **Funções**:
  - `eval_content_client(psg_node, client)` - função principal
  - `get_content_check_prompt(sentence)` - gera prompt
  - `chat_openai(prompt, client, try_number)` - chama GPT-4
- **Linhas**: 250-290
- **Modelo**: OpenAI GPT-4

#### Interpretação dos Valores

| Valor | Interpretação |
| :--- | :--- |
| 85 - 100 | Excelente - Cobertura e profundidade excepcionais |
| 70 - 84 | Muito bom - Boa cobertura e profundidade |
| 50 - 69 | Bom - Cobertura e profundidade adequadas |
| 30 - 49 | Fraco - Lacunas em cobertura ou profundidade |
| 0 - 29 | Muito fraco - Cobertura ou profundidade inadequadas |

#### Exemplo de Avaliação

**Critério: Topic Comprehensiveness (35 pontos)**

- GT cobre: Fondamentos, RNNs, CNNs, Vision Transformers, Aplicações
- LLM cobre: Fundamentos, CNNs, Applications
- **Score**: 20/35 (cobriu 3 de 5 tópicos principais)

**Critério: Depth (35 pontos)**

- GT explora cada tópico em 2-3 páginas
- LLM explora cada tópico em 1 parágrafo
- **Score**: 15/35 (superficial)

**Critério: Balance (30 pontos)**

- GT: distribuição equilibrada
- LLM: 60% em CNNs, 40% em outros
- **Score**: 18/30 (desbalanceado)

**Total: 53/100 (BOM)**

#### Relação com Comparação LLM vs GT

Content Quality mede a **visão completa do domínio** que o survey expressa. Um score alto indica que o LLM conseguiu capturar não apenas informações pontuais, mas uma compreensão holística e bem equilibrada do tópico.

---

## BERTScore (Bônus)

### O que é?

**BERTScore** é uma métrica moderna que usa embeddings do modelo BERT (Bidirectional Transformer) para avaliar **similaridade semântica** entre textos em nível mais profundo que n-grams.

Calcula:

- **Precision**: Qual fração de tokens do LLM têm correspondência semântica no GT?
- **Recall**: Qual fração de tokens do GT têm correspondência semântica no LLM?
- **F1**: Média harmônica de Precision e Recall

### Como é Usada?

```python
def calculate_bertscore_for_sections(text_llm, text_gt, model_type="roberta-large"):
    P, R, F1 = score([text_llm], [text_gt],
                     lang='en', model_type=model_type,
                     device='cuda')

    return {
        'precision': float(P[0]),
        'recall': float(R[0]),
        'f1': float(F1[0])
    }
```

### Localização no Código

- **Arquivo**: `src/rougeBleuFuncs.py`
- **Função**: `calculate_bertscore_for_sections(text_llm, text_gt, model_type)`
- **Linhas**: 83-117
- **Modelo**: roberta-large (via biblioteca `bert_score`)

### Interpretação

Valores de 0 a 1, onde:

- **Precision**: 0.90+ = LLM está focado em conceitos relevantes
- **Recall**: 0.90+ = LLM cobre a maioria do conteúdo do GT
- **F1**: 0.90+ = Excelente alinhamento semântico

---

## Exemplo de Saída JSON

Uma avaliação completa do sistema gera um JSON estruturado assim:

```json
{
  "survey_id": 26,
  "survey_title": "A Survey on Visual Transformer",
  "evaluation_results": {
    "Information_Collection": {
      "Comprehensiveness": {
        "Coverage": 0.75
      },
      "Relevance": {
        "Paper_Level": 0.82,
        "Section_Level": 0.78,
        "Sentence_Level": 0.71
      }
    },
    "Survey_Structure": {
      "Structure_Quality(LLM_as_judge)": 4,
      "SH-Recall": 0.86,
      "Subtitle_similarity": 0.68
    },
    "Survey_Content": {
      "Relevance": {
        "ROUGE-1": 0.72,
        "ROUGE-2": 0.58,
        "ROUGE-L": 0.65,
        "BLEU": 42.3
      },
      "Content LLM as a judge": 78
    }
  },
  "summary": {
    "average_information_collection": 0.77,
    "average_structure_quality": 0.85,
    "average_content_quality": 0.64,
    "overall_score": 0.75
  }
}
```

### Interpretação da Saída

- **Coverage 0.75**: LLM citou 75% dos artigos principais do GT
- **Paper_Level Relevance 0.82**: 82% das citações são relevantes globalmente
- **SH-Recall 0.86**: 86% dos títulos do GT foram recuperados semanticamente
- **Structure Quality 4**: GPT-4 deu nota 4/5 para estrutura
- **ROUGE-1 0.72**: 72% de overlap em palavras individuais
- **ROUGE-2 0.58**: 58% de overlap em pares de palavras (menos coesão)
- **ROUGE-L 0.65**: Sequência de palavras 65% similar
- **BLEU 42.3**: 42.3% de precisão em n-gramas
- **Content Quality 78**: GPT-4 deu nota 78/100 para conteúdo geral

---

## Conclusão

### Resumo das Métricas

As **13 métricas** (ou 10 principais + variações) foram organizadas em 3 categorias para avaliar diferentes aspectos da qualidade de surveys gerados por LLM:

#### 🔍 Information Collection (3 + 4 níveis)

- Avalia se o LLM recupera as referências certas do corpus
- Coverage (fração recuperada)
- 3 níveis de relevância (Paper → Section → Sentence)

#### 🏗️ Survey Structure (3)

- Avalia se a organização e estrutura são similares ao GT
- SH-Recall (recuperação semântica de títulos)
- Structure Quality (avaliação geral por LLM)
- Subtitle Similarity (proporção de títulos correspondentes)

#### 📄 Survey Content (6)

- Avalia se o conteúdo textual é similar semanticamente
- 4 métricas de overlap (ROUGE-1/2/L, BLEU)
- Content Quality (avaliação por LLM)
- BERTScore (similaridade semântica via embeddings)

### Como Usar esta Documentação

1. **Para entender uma métrica específica**: Encontre na seção apropriada
2. **Para interpretar resultados**: Veja a tabela de interpretação em cada métrica
3. **Para encontrar código**: Cada métrica cita arquivo e linha no repositório
4. **Para comparações LLM vs GT**: Verifique "Relação com Comparação" em cada métrica

### Valores de Referência

| Categoria | Métrica | Expected Range |
| :--- | :--- | :--- |
| Information | Coverage | 0.60 - 0.90 |
| Information | Paper Relevance | 0.70 - 0.95 |
| Information | Section Relevance | 0.60 - 0.90 |
| Information | Sentence Relevance | 0.50 - 0.85 |
| Structure | SH-Recall | 0.75 - 0.95 |
| Structure | Structure Quality | 3 - 5 |
| Structure | Subtitle Similarity | 0.40 - 0.80 |
| Content | ROUGE-1 | 0.60 - 0.85 |
| Content | ROUGE-2 | 0.40 - 0.70 |
| Content | ROUGE-L | 0.50 - 0.75 |
| Content | BLEU | 25 - 60 |
| Content | Content Quality | 60 - 90 |

### Próximos Passos

Para usar o sistema de avaliação completo:

1. Instale dependências: `pip install -r requirements.txt`
2. Prepare dados: Surveys em JSON e corpus de artigos
3. Configure modelos: Baixe embedding models (FlagEmbedding, CrossEncoder)
4. Configure API: OpenAI key para avaliação com GPT-4
5. Execute avaliação: Use `SurGEvaluator` com lista de métricas desejadas


---

## Diferença Rouge e Bleu

1) Por que BLEU é "oposto" ao ROUGE?

A diferença está na pergunta que cada métrica faz ao texto gerado pela IA.

    ROUGE (Recall-based) pergunta: "Do que estava no texto de referência, o quanto o modelo conseguiu lembrar (Recall)?"

    BLEU (Precision-based) pergunta: "Do que o modelo escreveu, o quanto está realmente correto e no texto de referência?"

Exemplo prático:
Imagine que o texto de referência (ground truth) é: "O gato preto dorme na cama"

    Modelo 1 (completo, mas prolixo): "Eu acho que o gato preto dorme na cama do meu vizinho".

    Modelo 2 (curto e direto): "O gato dorme".

O que cada métrica faria?

    ROUGE (Recall): Adoraria o Modelo 1, pois recuperou quase todas as palavras da referência (só adicionou coisas extras). O Modelo 2 seria punido por "esquecer" as palavras "preto", "na", "cama". ROUGE não liga se o texto ficou grande ou com lixo, desde que a informação da referência esteja lá.

    BLEU (Precision): Adoraria o Modelo 2, pois todas as palavras que gerou ("O", "gato", "dorme") estão na referência. O Modelo 1 seria punido por incluir "Eu", "acho", "que", "meu", "vizinho" – palavras que não estão no texto original. BLEU penaliza textos inflados, repetitivos ou com informações incorretas. É daí que vem a "brevity penalty": se o texto for curto demais (como um recall baixo), ele também toma um desconto, mas a base é a precisão.

Resumindo: é "oposto" porque um recompensa a cobertura completa (mesmo com lixo) e o outro recompensa a exatidão (mesmo que falte conteúdo).

---

**Documento gerado**: 23 de abril de 2026 | **Versão**: 1.0 | **Autor**: Sistema de Documentação automática