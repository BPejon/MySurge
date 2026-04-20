# Implementação: Extração Melhorada de Referências com Busca em Corpus

## Resumo

Foi implementado um sistema melhorado de extração de títulos de referências no formato journal com busca automática no corpus. Agora o algoritmo consegue:

1. **Detectar formatos de referência** - Identifica se uma referência segue o padrão `[num] Autores. Título. *Revista*, DOI`
2. **Extrair títulos corretamente** - Mesmo com iniciais de autores (P. Connolly, etc) e quebras de linha
3. **Buscar no corpus** - Procura o título no `corpusMatScience.json` 
4. **Retornar resultado** - Em formato `"num:doc_id"` se encontrar, ou `"num:título"` se não encontrar

## Arquivos Modificados

### [`src/informationFuncs.py`](src/informationFuncs.py)

Adicionadas as seguintes funções:

#### `load_corpus_index(corpus_path)`
- Carrega o corpus em cache global
- Retorna dicionário mapeando títulos (lowercase) → doc_id
- Evita múltiplas leituras do arquivo JSON

#### `is_journal_format(reference_text)`
- Detecta padrão: `[num] ... . ... . *`
- Retorna `True` se referência segue formato journal
- Retorna `False` para outros formatos

#### `extract_title_from_reference(reference_text)`
- **Problema resolvido**: Extrai corretamente mesmo com iniciais (P. Connolly)
- Usa regex: `and\s+\w+\.\s+(.+?)\.\s*\*`
- Remove sobrenome duplicado no início do resultado
- Limpa quebras de linha e espaços extras
- Retorna apenas o título

#### `find_title_in_corpus(title, corpus_path)`
- Busca exata (case-insensitive) primeiro
- Se não encontrar, faz busca similar (threshold: 85% de similaridade)
- Retorna `doc_id` se encontrar, `None` caso contrário

#### `extract_references_improved(text, corpus_path)`
- Função principal nova
- Extrai todas as referências (`[num]`)
- Para cada referência:
  - Detecta formato (journal ou outro)
  - Se journal: extrai título + busca corpus + retorna `"num:doc_id"` ou `"num:título"`
  - Se outro: usa método antigo (extrai sentença)

## Exemplo de Uso

```python
from informationFuncs import extract_references_improved

text = """
## References

[1] Karem Al-Garadi, Ammar El-Hussein, Mahmoud Elsayed, P. Connolly, Mohamed Mahmoud, M. Johns, and A. Adebayo. A rock core wettability index using NMR T2 measurements. *Journal of Petroleum Science and Engineering*, 208:109386, 2021.

[2] Mohammad Albusairi and Carlos Torres-Verdin. Rapid modeling of borehole measurements. *Geophysics*, May 2021.
"""

results = extract_references_improved(text, "dataMatScience/corpusMatScience.json")

for ref_num, content in results:
    print(f"[{ref_num}] -> {content}")
```

**Output esperado:**
```
[1] -> 1:9
[2] -> [2] Mohammad Albusairi and Carlos Torres-Verdin. Rapid modeling of borehole measurements...
```

## Especificação de Responsabilidade

| Função | Responsabilidade |
|--------|------------------|
| `load_corpus_index()` | Carregar e cachear o índice do corpus |
| `is_journal_format()` | Detectar padrão `[num] Autores. Título. *Revista*` |
| `extract_title_from_reference()` | Extrair apenas o título corretamente |
| `find_title_in_corpus()` | Buscar título no corpus (exata + similar) |
| `extract_references_improved()` | Orquestrar todo o fluxo |

## Compatibilidade

- ✅ A função original `extract_references()` continua intacta
- ✅ `extract_cites_with_subtitle_and_sentence()` continua funcionando
- ✅ Pode ser usada em paralelo com a função antiga
- ✅ Suporta múltiplos formatos de referência (não quebra para formatos não-journal)

## Teste de Validação

Execute o script de teste para validar funcionamento:

```bash
python3 test_reference_extraction.py
```

**Resultado esperado:**
```
================================================================================
TODOS OS TESTES PASSARAM! ✓
================================================================================
```

## Próximas Etapas

1. Integrar `extract_references_improved()` no lugar de `extract_references()` em `extract_cites_with_subtitle_and_sentence()`
2. Atualizar as funções no `evaluator.py` que usam referências
3. Testar com o arquivo `0.md` completo
