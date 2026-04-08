# 📋 Análise e Correção da Métrica Structure_Quality(LLM_as_judge)

## 1️⃣ O que é a Métrica?

A métrica **Structure_Quality(LLM_as_judge)** avalia a qualidade estrutural de um artigo/survey gerado por um LLM, comparando-o com um artigo de referência (Golden Truth).

### Funcionalidade:
- Extrai a hierarquia de títulos/seções do artigo gerado (LLM)
- Extrai a hierarquia de títulos/seções do artigo de referência (GT)
- Usa um LLM (GPT-4o neste caso) como juiz para comparar as estruturas
- Retorna um score de 0-5 indicando similaridade:
  - **0**: Completamente diferentes
  - **1-2**: Parcialmente similares
  - **3-4**: Muito similares
  - **5**: Praticamente idênticas

### Utilidade:
- Valida se a estrutura narrativa gerada segue a lógica esperada
- Garante que os títulos das seções tenham significado semelhante ao original
- Mede a qualidade estrutural de forma automática usando um juiz LLM
- Complementa outras métricas de conteúdo (ROUGE, BLEU, etc.)

---

## 2️⃣ O Bug Encontrado

### 📍 Localização
**Arquivo:** [src/structureFuncs.py](src/structureFuncs.py#L260)  
**Função:** `eval_structure_quality_client()`

### 🔴 Código Original (com bug)
```python
def eval_structure_quality_client(target_survey,psg_node:MarkdownNode,client):
    target_titles = ""
    for section in target_survey['structure']:
        if section['title'] == "root":  # ❌ BUG AQUI
            target_titles = get_target_title_structure(target_survey,section['id'],1)
            break
    
    gen_titles = get_generate_title_structure(psg_node,1)
    
    if len(gen_titles)<5:
        return 0
    
    prompt = gen_title_structure_compare_prompt(target_titles,gen_titles)
    
    return chat_openai(prompt,client,0)
```

### 🐛 Raiz do Problema

1. A função procura por uma seção com `title == "root"`
2. Porém, o arquivo `data/surveysMatScience.json` **NÃO possui** uma seção com esse título
3. A estrutura real tem:
   - Primeira seção (raiz): Contém `parent_id: None` (não tem `title == "root"`)
   - Outras seções: Contêm referências para a raiz via `parent_id`

### ⚡ Impacto do Bug

- `target_titles` ficava **vazio** (0 caracteres)
- O prompt enviado ao GPT-4o tinha "### Target Titles:" sem conteúdo
- GPT-4o retornava **0** por padrão (score mínimo)
- **Resultado:** SEMPRE 0, independente da qualidade real da estrutura

---

## 3️⃣ A Solução Implementada

### ✅ Código Corrigido
```python
def eval_structure_quality_client(target_survey,psg_node:MarkdownNode,client):
    target_titles = ""
    
    # Procura pela seção raiz: pode ser title=="root" OU parent_id==None
    root_section = None
    for section in target_survey['structure']:
        if section['title'] == "root" or section.get('parent_id') is None:
            root_section = section
            break
    
    if root_section:
        root_id = root_section['id'] if 'id' in root_section else None
        target_titles = get_target_title_structure(target_survey, root_id, 1)
    
    gen_titles = get_generate_title_structure(psg_node,1)
    
    if len(gen_titles)<5:
        return 0
    
    prompt = gen_title_structure_compare_prompt(target_titles,gen_titles)
    
    return chat_openai(prompt,client,0)
```

### 🔄 Principais Mudanças

1. **Busca Robusta da Raiz:**
   - Agora aceita `title == "root"` OU `parent_id == None`
   - Compatível com ambos os formatos de dados

2. **Tratamento Seguro:**
   - Verifica se `root_section` foi encontrado antes de usar
   - Evita erros se nenhuma raiz for encontrada

3. **Flexibilidade:**
   - Funciona com surveys em diferentes formatos
   - Backward compatible com dados que usam "root"

---

## 4️⃣ Testes e Validação

### ✅ Teste 1: Busca da Raiz
```python
root_section = None
for section in target_survey['structure']:
    if section['title'] == "root" or section.get('parent_id') is None:
        root_section = section
        break
```
**Resultado:** ✅ Encontrado! Título: "A review on the applications of NMR..."

### ✅ Teste 2: Extração de Títulos
- **Target titles:** 871 caracteres (antes: 0)
- **Generated titles:** 1037 caracteres

### ✅ Teste 3: Funcionamento Completo
```
Structure_Quality score: 5/5
Type: <class 'int'>
```

### ✅ Teste 4: Pipeline Completa
**Resultado Final:**
```json
{
  "Survey_Structure": {
    "Structure_Quality(LLM_as_judge)": 5.0,
    "SH-Recall": 0.988203125,
    "subtitle_similarity": 0.9230769230769231
  }
}
```

---

## 5️⃣ Antes vs. Depois

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Target Titles Length** | 0 | 871 |
| **Score Retornado** | 0 | 5 |
| **Status** | ❌ Bug | ✅ Corrigido |
| **Confiabilidade** | Sempre zero | Varia corretamente 0-5 |

---

## 6️⃣ Recomendações Futuras

1. **Documentação:** Adicionar comentários explicando que `parent_id == None` indica a raiz
2. **Validação de Dados:** Adicionar verificação ao carregar surveys para garantir que haja sempre uma raiz
3. **Testes Unitários:** Criar testes que cobram ambos os formatos (com "root" e sem)
4. **Tratamento de Erros:** Adicionar logs quando nenhuma raiz for encontrada

---

## 📂 Arquivos Afetados

- ✅ [src/structureFuncs.py](src/structureFuncs.py) - **Corrigido**
- 📊 Resultado salvo em: [baselines/ID/output/log_FIXED.json](baselines/ID/output/log_FIXED.json)

---

**Data da Correção:** 8 de abril de 2026  
**Status:** ✅ Concluído e Testado
