import re
from markdownParser import *
import numpy as np
from scipy.spatial.distance import pdist

import gc

def eval_coverage(target_cites:list, gen_cite_map:dict):
    print(f"target cites: {target_cites}")
    print("*****")
    print(f"generated cites: {gen_cite_map}")
    gen_cites = []
    for k,v in gen_cite_map.items():
        gen_cites.append(v)
        
    all = len(target_cites)
    hit = 0
    print("*****")
    print(f"gen cites: {gen_cites}")
    for c in target_cites:
        if c in gen_cites:
            hit += 1

    print(f"hit: {hit}, all: {all}, coverage: {hit/all}")
            
    return hit/all

def eval_relevance_paper(target_survey,gen_cite_map:dict,cite_content:dict,nli_model):
    #print("target_survey: {target_survey}")
    #print(f"Dict {gen_cite_map}")
    #print(f"Cite content {cite_content}")
    #print(f"nli_model {nli_model}")
    if len(gen_cite_map) == 0:
        return 0
    
    all = len(gen_cite_map)
    hit = 0
    nli_pairs = []
    for k,v in gen_cite_map.items():
        if v in target_survey['all_cites']:
            hit += 1
        else:
            if cite_content[k][0] != "[NOTEXIST]":
                nli_pairs.append(cite_content[k])
    
    if len(nli_pairs) > 0:  
        print("Eval relevanve paper")
        print(gen_cite_map)
        print(nli_pairs)
        
        scores = nli_model.predict(nli_pairs)
        
        label_mapping = ['contradiction', 'entailment', 'neutral']
        # 计算 entailment 的 Softmax 值
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        # print("Papaer possibilities")
        # print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
    return hit/all

def eval_relevance_section(nli_pairs_origin,nli_model):
    if len(nli_pairs_origin) == 0:
        return 0
    
    missed = 0
    nli_pairs = []
    for citation, subtitle in nli_pairs_origin:
        if citation != "[NOTEXIST]":
            nli_pairs.append((subtitle,citation))
        else:
            missed += 1
    #print("Eval relevanve section")
    #print(nli_pairs)
    
    
    all = len(nli_pairs) + missed
    hit = 0
    if len(nli_pairs) > 0:
        scores = nli_model.predict(nli_pairs)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        #print("section possibilities")
        #print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
    return hit/all

def eval_relevance_sentence(nli_pairs_origin,nli_model):
    if len(nli_pairs_origin) == 0:
        return 0
    
    missed = 0
    nli_pairs = []
    
    for citation, sentence in nli_pairs_origin:
        if "[NOTEXIST]" != citation:
            nli_pairs.append((sentence,citation))        
        else: 
            missed += 1
            
    #print("Eval relevanve sentence")
    #print(nli_pairs)
    
    all = len(nli_pairs) + missed
    hit = 0
    if len(nli_pairs) > 0:
        scores = nli_model.predict(nli_pairs)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        # 计算 entailment 的 Softmax 值
        # values = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        # entailment_probs = values[:,1].tolist()
        # print("sentence possibilities")
        # print(entailment_probs)
        # for i in entailment_probs:
        #     if i > 0.6:
        #         hit += 1
        
        for c,e,n in scores:
            if e > c and e > n:
                hit += 1
            elif n > c and n > e:
                if e > c:
                    hit += 0.5
        
    return hit/all
    
    
def extract_references(text):
    pattern = r'\[(\d+)\]'
    matches = re.finditer(pattern, text)
    results = []
    
    for match in matches:
        ref_number = int(match.group(1))
        start_pos = match.start()
        sentence_start = text.rfind('.', 0, start_pos) + 1
        sentence_end = text.find('.', start_pos)
        if sentence_end == -1:
            sentence_end = len(text)
        sentence = text[sentence_start:sentence_end].strip()
        results.append((ref_number, sentence))
    
    return results

def extract_cites_with_subtitle_and_sentence(psg_node:MarkdownNode):
    res = []
    text = "\n".join(psg_node.content)
    tmp_res = extract_references(text)
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        for ref_number, sentence in tmp_res:
            res.append((ref_number, psg_node.title, sentence))
    
    for child in psg_node.children:
        res.extend(extract_cites_with_subtitle_and_sentence(child))
    
    return res


def get_content_list(psg_node:MarkdownNode):
    res = []
    text = "\n".join(psg_node.content)
    if len(text) >= 100:
        res.append(text)
    for child in psg_node.children:
        tmp_res = get_content_list(child)
        res.extend(tmp_res)
        
    return res

def get_content_check_prompt(sentence):
    print(f"Check content: {sentence}")
    prompt = f"""
**Task:** As an expert literature review evaluator, assess only the **coverage quality** of a generated literature review 
**Coverage Quality Definition:**
Coverage quality refers to the comprehensiveness, depth, and balance of topic treatment within a literature review, including the breadth of relevant concepts covered and the proportional attention given to each area.

Generated Review for Evaluation:
---
{sentence}
---
**Coverage Evaluation Criteria (100 points total):**
1. **Topic Comprehensiveness (35 points)** 2. **Discussion Depth (35 points)**
- Range of essential topics covered - Detail level of concept analysis
- Inclusion of emerging areas - Development of key arguments
- Identification of key concepts - Thoroughness of explanations
Scoring Guide: Scoring Guide:
- 30-35: Comprehensive coverage with emerging topics - 30-35: Exceptional depth across topics
- 20-29: Good coverage with minor gaps - 20-29: Good depth with some variation
- 0-19: Significant omissions or major gaps - 0-19: Consistently superficial treatment
3. **Content Balance (30 points)**
- Proportional coverage of topics
- Appropriate emphasis distribution
- Logical allocation of space
Scoring Guide:
- 25-30: Well-balanced coverage throughout
- 15-24: Generally balanced with minor issues
- 0-14: Significant imbalance issues
**Scoring Requirements:**
- Prioritize accuracy over conservatism
- AVOID "safe" middle-range scores that don't reflect true quality. Score based purely on merit, not on scoring "comfort zones"
- Each score must reflect precise performance level, not range averages (e.g., 25 for 20-29 range)
- Use full scoring range (0-100)
- Base scores on objective comparison to human reference
- Acknowledge that best practices may evolve

**Output Format:**
Return only a single numerical score (0-100). No additional commentary.


"""
    return prompt


def chat_openai(prompt, client,try_number):
    if try_number == 5:
        print("Failed to get valid response after 5 tries.")
        return None
    #print(f"Try {try_number} time")
    #print("Query:")
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
        if not (response.choices[0].message.content.isdigit() and 0 <= int(response.choices[0].message.content) <= 100):
            ans =  chat_openai(prompt,client,try_number + 1)
        else :
            ans = int(response.choices[0].message.content)
        #result_queue.put(ans)
    except Exception as e:
        print(f"An error occurred: {e}")
        return chat_openai(prompt,client,try_number + 1)
        
    return ans

    
def eval_content_client(psg_node: MarkdownNode, client):
    psgs = get_content_list(psg_node)
    
    # Concatenar todos os conteúdos em um documento único
    full_article = "\n\n".join(psgs)
    
    # Limitar o tamanho total do artigo se necessário
    if len(full_article) > 5000:
        full_article = full_article[:5000]
        if full_article[-1] != '.':
            full_article = full_article[:full_article.rfind('.')]
    
    # Avaliar o artigo inteiro uma única vez
    score = chat_openai(get_content_check_prompt(full_article), client, 0)
    
    return score


# def chat(text,model,tokenizer,try_number):
#     if try_number == 5:
#         print("Failed to get valid response after 5 tries.")
#         return None
    
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
    

# def eval_logic(psg_node: MarkdownNode, model, tokenizer):
#     psgs = get_content_list(psg_node)
#     np.random.seed(42)
#     psgs = np.random.choice(psgs, 20, replace=False)
    
#     result = 0
#     for i in range(len(psgs)):
#         if len(psgs[i]) > 1000:
#             psgs[i] = psgs[i][:1000]
#             if psgs[i][-1] != '.':
#                 psgs[i] = psgs[i][:psgs[i].rfind('.')]
                
#         prompt = get_logic_check_prompt(psgs[i])
#         result += chat(prompt,model,tokenizer,0)
    
#     return result/len(psgs)
        
      
    