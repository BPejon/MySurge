import re
from markdownParser import *

import os 
import gc
import numpy as np

def calc_sim(A, B, model):
    embedding_A = model.encode(A)  # Shape: (len(A), embedding_dim)
    embedding_B = model.encode(B)  # Shape: (len(B), embedding_dim)

    norm_A = np.linalg.norm(embedding_A, axis=1, keepdims=True)  # Shape: (len(A), 1)
    norm_B = np.linalg.norm(embedding_B, axis=1, keepdims=True)  # Shape: (len(B), 1)


    similarity_matrix = np.dot(embedding_A, embedding_B.T) / (norm_A * norm_B.T)

    return similarity_matrix

def normalize_sims(sims):
    return sims


def soft_heading_recall(G, P, model):
    """
    Calcula Soft Heading Recall entre títulos alvo (G) e gerados (P).
    
    SH-Recall mede a fração dos títulos alvo que foram "recuperados" nos títulos
    gerados, considerando similaridade semântica (não apenas matches exatos).
    
    A fórmula: SH-Recall = sum(max_similarity_com_P[i]) / |G|
    Para cada título alvo G[i], encontra o título gerado P[j] mais similar,
    e retorna a média dessa máxima similaridade como indicador de recuperação.
    
    Args:
        G: Lista de títulos alvo (referência/ground truth)
        P: Lista de títulos gerados
        model: Modelo de embedding para calcular similaridades
        
    Returns:
        float: SH-Recall entre 0 e 1
    """
    
    # Se não há títulos alvo, não há nada para recuperar
    if len(G) == 0:
        return 0
    
    # Se não há títulos gerados, nada foi recuperado
    if len(P) == 0:
        return 0
    
    # Calcula matriz de similaridades: |G| x |P|
    # sims[i][j] = similaridade entre G[i] e P[j]
    sims = calc_sim(G, P, model)  # shape: (len(G), len(P))
    
    # Para cada título alvo, encontra a máxima similaridade com qualquer título gerado
    # Isto representa "foi recuperado?"
    max_similarities = []
    for i in range(len(G)):
        max_sim = max(sims[i]) if len(P) > 0 else 0
        max_similarities.append(max_sim)
    
    # SH-Recall é a média das máximas similaridades
    # Isto responde: "Em média, qual é a similaridade do melhor match para cada título alvo?"
    soft_recall = sum(max_similarities) / len(G) if len(G) > 0 else 0
    
    return soft_recall


def get_title_list(psg_node:MarkdownNode):
    res = []
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        res.append(psg_node.title[:512])
    for child in psg_node.children:
        tmp_res = get_title_list(child)
        res.extend(tmp_res)
        
    return res

def eval_SHRecall(target_survey,psg_node: MarkdownNode,model):
    
    target_titles = []
    for section in target_survey['structure']:
        if len(section['content']) < 100:
            continue
        target_titles.append(section['title'])
        #subtitles_map.append(section['title'])
        
    gen_titles = get_title_list(psg_node)

    #print("Target Titles:",target_titles)
    #print("----------------")
   # print("Generated Titles:",gen_titles)

    
    if len(gen_titles) == 0:
        return 0
    
    return soft_heading_recall(target_titles,gen_titles,model)
    
    
def get_target_title_structure(target_survey,id,level):
    res = ""
    for section in target_survey['structure']:
        if section['parent_id'] == id:
            res += "#"*level + " " + section['title'] + "\n"
            res += get_target_title_structure(target_survey,section['id'],level+1)
            res += '\n'
    return res   

def get_generate_title_structure(psg_node:MarkdownNode,level):
    res = ""
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        res += "#"*level + " " + psg_node.title + "\n"
    for child in psg_node.children:
        tmp_res = get_generate_title_structure(child,level+1)
        res += tmp_res
    return res
    
    
def gen_title_structure_compare_prompt(target_titles, generated_titles):
    prompt = f"""You are an AI evaluator. Your task is to compare the generated titles with the target titles and assign a score from 0 to 5 based on their similarity in structure, meaning, and wording.

### Target Titles:
{target_titles}

### Generated Titles:
{generated_titles}

## **Scoring Criteria:**

- **0 – Completely Different:**  
  - Nearly no words in common.  
  - Completely different meanings.  
  - No similarity in structure or phrasing.  

- **1 – Somewhat Different:**  
  - Few words overlap, but they are not key terms.  
  - The meaning is somewhat related but mostly different.  
  - The sentence structures are significantly different.  

- **2 – Somewhat Similar:**  
  - Some key words are shared, but others are different.  
  - The general topic is the same, but the emphasis may differ.  
  - The sentence structures are different but not entirely unrelated.  

- **3 – Similar:**  
  - Several key words are shared.  
  - The meaning is largely the same with slight variations.  
  - The structure is somewhat similar, but there may be word substitutions.  

- **4 – Very Similar:**  
  - Most key words match.  
  - The meaning is nearly identical.  
  - The phrasing and structure are very close, with minor rewording.  

- **5 – Almost Identical:**  
  - Nearly all key words match exactly.  
  - The meaning is fully preserved.  
  - The phrasing and structure are identical or differ only in trivial ways.  

### **Instructions:**  
Analyze the generated titles based on the criteria above and provide a single score between 0 and 5.  
**Output only the score as a number, without any additional explanation or comments.**
"""
    return prompt

# def chat(text,model,tokenizer,try_number):
#     if try_number == 5:
#         print("Failed to get valid response after 5 tries.")
#         return None
#     #print("Query:")
#     #print(text)
#     prompt = text
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     # print("TOKENLEN",len(model_inputs['input_ids'][0]))

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=16
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     #print("Answer:")
#     #print(response)

#     response = response.strip('.')
    
#     ans = None
    
#     if not re.match(r'^[0-5]$', response):
#         ans =  chat(text,model,tokenizer,try_number + 1)
#     else :
#         ans = int(response)
    
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return ans
    
    
    
# def eval_structure_quality(target_survey,psg_node:MarkdownNode,model,tokenizer):
#     target_titles = ""
#     for section in target_survey['structure']:
#         if section['title'] == "root":
#             target_titles = get_target_title_structure(target_survey,section['id'],1)
#             break
    
#     gen_titles = get_generate_title_structure(psg_node,1)
    
#     if len(gen_titles)<5:
#         return 0
    
#     prompt = gen_title_structure_compare_prompt(target_titles,gen_titles)
    
#     return chat(prompt,model,tokenizer,0)
    
def chat_openai(prompt, client, try_number):
    if try_number == 5:
        print("Failed to get valid response after 5 tries.")
    #print(f"Try {try_number} time")
    #print(prompt)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=100
        )
        #print(f"Answer:{response.choices[0].message.content}")
        ans = None
        if not re.match(r'^[0-5]$', response.choices[0].message.content):
            ans =  chat_openai(prompt,client,try_number + 1)
        else :
            ans = int(response.choices[0].message.content)
        return ans
    except Exception as e:
        print(f"An error occurred: {e}")
        return chat_openai(prompt,client,try_number + 1)    
    

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

def extract_section_text(node: MarkdownNode) -> str:
    """
    Extrai o texto de uma seção/subseção (apenas do nó atual, sem filhos).
    
    Args:
        node: MarkdownNode da seção a extrair
        
    Returns:
        String com o conteúdo completo da seção
    """
    if node is None:
        return ""
    
    # Concatena todas as linhas de conteúdo do nó
    text = "\n".join(node.content) if node.content else ""
    
    # Remove espaços em branco extras
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove linhas em branco múltiplas
    text = text.strip()
    
    return text


def calculate_content_metrics(text_llm: str, text_gt: str) -> dict:
    """
    Calcula métricas de similaridade de conteúdo entre dois textos.
    
    Args:
        text_llm: Texto da seção gerada pela LLM
        text_gt: Texto da seção do Ground Truth
        
    Returns:
        dict com BERTScore e ROUGE-L
    """
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    
    metrics = {
        'bertscore_f1': 0.0,
        'rouge_l': 0.0
    }
    
    # Trata caso onde textos são muito curtos
    if len(text_llm) < 100 or len(text_gt) < 100:
        return metrics
    
    try:
        # Calcula BERTScore usando modelo padrão (roberta-large)
        # Não especificamos model_type para usar o padrão 'bert-base-uncased'
        P, R, F1 = bert_score([text_llm], [text_gt], lang='en')
        metrics['bertscore_f1'] = float(F1[0])
    except Exception as e:
        print(f"Erro ao calcular BERTScore: {e}")
        metrics['bertscore_f1'] = 0.0
    
    try:
        # Calcula ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(text_gt, text_llm)
        metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Erro ao calcular ROUGE-L: {e}")
        metrics['rouge_l'] = 0.0
    
    return metrics
    