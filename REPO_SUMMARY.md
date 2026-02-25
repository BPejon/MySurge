# Resumo do Repositório SurGE

Este documento resume a finalidade, estrutura e fluxo principal do repositório SurGE (Survey Generation benchmark & evaluation).

**Overview**
- **Objetivo**: Fornecer um dataset e uma framework automática para avaliar sistemas de geração de surveys científicos (recuperação, organização e síntese).
- **Dados principais**: `data/surveys.json` (ground-truth surveys) e `data/corpus.json` (knowledge base de artigos).

**Arquivos Principais**
- **`README.md`**: Guia de alto nível, instruções de instalação e uso rápido.
- **`requirements.txt`**: Dependências Python necessárias.
- **`src/download.py`**: Script para baixar `corpus.json` e `surveys.json` (usa `gdown`).
- **`src/test_final.py`**: Script de linha de comando que inicializa o avaliador e executa avaliações em diretórios de saídas geradas.
- **`src/evaluator.py`**: Classe `SurGEvaluator` — carga dados, configura clientes (OpenAI / modelos locais) e orquestra avaliações.
- **`src/markdownParser.py`**: Parser simples que converte um arquivo `.md` gerado em uma árvore (`MarkdownNode`) e extrai referências.
- **`src/rougeBleuFuncs.py`**: Cálculo de ROUGE e BLEU entre conteúdo gerado e ground-truth.
- **`src/structureFuncs.py`**: Avaliação da qualidade de estrutura (SH-Recall, prompt para avaliador LLM) e utilitários de comparação de títulos/estruturas.
- **`src/informationFuncs.py`**: Métricas de cobertura, relevância (paper/section/sentence) e checagem lógica (usa CrossEncoder e chamadas a LLMs).

**Fluxo de Execução (resumido)**
- Preparar ambiente: criar venv/conda e `pip install -r requirements.txt`.
- Baixar dados: `python src/download.py` (salva em `data/`).
- Preparar saídas geradas: cada survey gerado deve ficar em `./<passage_dir>/<survey_id>/` contendo um `.md` com referências no formato esperado.
- Rodar avaliação: exemplo

```
python src/test_final.py --passage_dir ./baselines/ID/output --save_path ./baselines/ID/output/log.json --device 0 --api_key sk-xxx
```

**Como o avaliador trabalha (componentes & interações)**
- **Leitura dos dados**: `SurGEvaluator` carrega `data/surveys.json` e `data/corpus.json` em mapas para acesso rápido.
- **Parsing do `.md`**: cada saída gerada é parseada por `markdownParser.parse_markdown` (arvore) e `parse_refs` (mapeia `[n] Title`).
- **Mapeamento de referências**: o avaliador tenta mapear cada referência textual para `doc_id` via `title2docid` (normaliza letras).
- **Métricas**:
  - ROUGE / BLEU: `rougeBleuFuncs.eval_rougeBleu` compara trechos de conteúdo.
  - SH-Recall (estrutura): `structureFuncs.eval_SHRecall` usa embeddings (FlagEmbedding) para avaliar recuperação de headings.
  - Relevância/Coverage: `informationFuncs` calcula coverage e usa `CrossEncoder` (NLI) para avaliar relevância de paper/section/sentença.
  - Structure_Quality / Logic: prompts enviados a LLMs (por padrão está configurado para OpenAI via `OpenAI` client) para julgamento qualitativo.

**Principais dependências e observações**
- **Modelos / APIs**: código suporta combinação local (CrossEncoder, FlagEmbedding) e LLM via OpenAI client. Para usar OpenAI, passe `--api_key` (ou ajuste `SurGEvaluator` internamente).
- **Dependências pesadas**: `torch`, `FlagEmbedding`, `bertopic` e modelos de embedding/julgador podem requerer GPU e versões específicas de CUDA.
- **Formato esperado das saídas**: o `.md` gerado precisa conter headings para estruturar o survey e referências no formato `[n] Title` (ou `[n] Author. *Title* ...`).

**Métricas suportadas (opções `--eval_list`)**
- **ALL**: todas as métricas abaixo.
- **ROUGE-BLEU**: ROUGE-1/2/L e BLEU entre textos gerados e ground-truth.
- **SH-Recall**: soft heading recall entre títulos alvo e gerados.
- **Structure_Quality**: julgamento de estrutura via LLM.
- **Coverage**: fração de citações alvo recuperadas.
- **Relevance-Paper / Relevance-Section / Relevance-Sentence**: avaliações de relevância usando CrossEncoder NLI.
- **Logic**: avaliação da coerência lógica via LLM.

**Exemplos úteis**
- Baixar dados: `python src/download.py`.
- Rodar avaliação para um baseline: ver exemplo de `README.md` (usar `--api_key` se Structure_Quality/Logic via OpenAI).

**Limitações & notas práticas**
- Para avaliações que usam OpenAI, verifique limites de tokens e custos — prompts para Structure_Quality e Logic podem enviar trechos longos.
- `markdownParser.parse_refs` é simples e pode falhar em formatos de referência muito distintos; padronize as referências nos `.md` gerados.
- Mapeamento `title2docid` depende de normalização por letras — títulos muito diferentes não serão automaticamente reconhecidos.

**Próximos passos sugeridos**
- Rodar `python src/download.py` e verificar `data/` (se quiser, posso executar esse passo para você).
- Testar a avaliação em um pequeno conjunto de surveys gerados (ex.: 1 ou 2 ids) para validar o pipeline e as chaves de API.
- Melhorias possíveis: robustecer o parser de referências, adicionar heurísticas de fuzzy-match para títulos, e instrumentar um modo offline para `Structure_Quality` (modelo local).

---

Arquivo gerado automaticamente: `REPO_SUMMARY.md` — diga se deseja que eu detalhe qualquer seção ou rode os scripts de download/avaliação agora.
