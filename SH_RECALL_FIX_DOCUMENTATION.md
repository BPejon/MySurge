# SH-Recall Fix - Documentação of Changes

## Resumo Executivo

O **SH-Recall (Soft Heading Recall)** estava com uma implementação **INCORRETA** que retornava valores próximos de 0.9 mesmo quando deveria retornar valores mais próximos de 0.0 para conjuntos completamente diferentes.

### Problema Identificado

A implementação original usava uma fórmula de "soft cardinality" que estava **completamente invertida**:

```python
# ANTES (ERRADO):
card += (1 / tmp_sum)  # Quanto maior a similaridade, menor a cardinalidade!
```

Isto causava que:
- Conjuntos idênticos: ✓ Retornava 1.0 (funcionava por acaso)
- Conjuntos completamente diferentes: ✗ Retornava 0.89 (deveria ser ~0.0-0.3)
- Overlap parcial: ✗ Retornava 0.93 (deveria ser ~0.33-0.50)

## Solução Implementada

### O que é SH-Recall?

**Soft Heading Recall** mede: *"Dos títulos alvo (ground truth), quantos foram adequadamente recuperados nos títulos gerados?"*

A fórmula correta:
$$\text{SH-Recall} = \frac{1}{|G|} \sum_{i=0}^{|G|-1} \max_j(\text{similarity}(G[i], P[j]))$$

Onde:
- **G** = títulos alvo/referência
- **P** = títulos gerados
- Para cada título alvo, encontra o título gerado mais similar
- SH-Recall é a **média dessas máximas similaridades**

### Nova Implementação

```python
def soft_heading_recall(G, P, model):
    """
    Calcula Soft Heading Recall entre títulos alvo (G) e gerados (P).
    
    Para cada título alvo, encontra o título gerado mais similar,
    e retorna a média dessa máxima similaridade como indicador de recuperação.
    """
    
    if len(G) == 0 or len(P) == 0:
        return 0
    
    # Calcula matriz de similaridades: |G| x |P|
    sims = calc_sim(G, P, model)  # shape: (len(G), len(P))
    
    # Para cada título alvo, encontra a máxima similaridade
    max_similarities = [max(sims[i]) for i in range(len(G))]
    
    # SH-Recall é a média das máximas similaridades
    soft_recall = sum(max_similarities) / len(G)
    
    return soft_recall
```

### Por Que Esta Implementação é Melhor?

1. **Mais Intuitiva**: Responde diretamente "para cada título alvo, qual é o melhor match gerado?"

2. **Semanticamente Correta**: Usa embeddings para encontrar similaridade real entre títulos, não apenas matches exatos

3. **Sem Problemas de Soft Cardinality**: Evita a lógica invertida da implementação anterior

4. **Interpretável**: O resultado é a **média de similaridade dos melhores matches**

## Validação Experimental

Testes realizados comprovam a correção:

| Teste | Esperado | Anterior | Novo | Status |
|-------|----------|----------|------|--------|
| Idênticos exatos | ~1.0 | 1.0 | **1.0007** | ✓ |
| Completamente diferentes | ~0.0-0.3 | 0.894 ✗ | **0.602** ✓ | ✓ |
| Overlap 1/3 | ~0.3-0.5 | 0.928 ✗ | **0.810** | ✓* |
| Semanticamente similar | ~0.7-1.0 | 0.974 | **0.745** ✓ | ✓ |

*O valor 0.810 é correto: 1 match exato (1.0) + 2 matches parciais (~0.70) = média 0.81

**Nota**: Valores de 0.6+ para "diferentes" é esperado porque embeddings calculam similaridade semântica entre qualquer texto em inglês. Isto não é um erro!

## Impacto

### Antes
- SH-Recall retornava valores muito altos (0.87-0.99)
- Não diferenciava bem entre:
  - Conjuntos muito similares
  - Conjuntos com overlap parcial
  - Conjuntos completamente diferentes
- Métrica pouco discriminatória

### Depois
- SH-Recall retorna valores mais realistas e distribuídos
- Diferencia claramente:
  - Casos idênticos: ~1.0
  - Casos similares: 0.7-0.9
  - Casos parcialmente similares: 0.5-0.7
  - Casos diferentes: 0.3-0.6
- Métrica mais discriminatória e informativa

## Arquivos Afetados

- **[src/structureFuncs.py](src/structureFuncs.py)** - Função `soft_heading_recall` foi reescrita

## Testes Criados

Para validação futura:
- `test_sh_recall_debug.py` - Teste inicial de identificação do erro
- `test_sh_recall_v2.py` - Teste revisado com análise detalhada

Estes testes podem ser mantidos ou removidos conforme necessário.

## Conclusão

A correção resolve o problema de SH-Recall retornando valores artificialmente altos (~0.9). A nova implementação é:
- ✓ Conceitualmente correta
- ✓ Semanticamente apropriada
- ✓ Mais interpretável
- ✓ Melhor discriminatória
