import json
import argparse
import time
import re
import markdownParser,rougeBleuFuncs,structureFuncs,informationFuncs
import os
from sentence_transformers import CrossEncoder
# from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from FlagEmbedding import FlagModel
from openai import OpenAI
import httpx
import numpy as np
from scipy.spatial.distance import cdist

def normalize_string(s):
        """
        Normaliza uma string para comparação, mantendo a informação importante.
        Remove pontuação especial e converte para minúsculas, mas preserva estrutura.
        """
        # Converte para minúscula
        s = s.lower()
        # Remove caracteres especiais mas mantém letras, números e espaços
        s = re.sub(r'[^\w\s]', ' ', s)
        # Remove espaços múltiplos
        s = re.sub(r'\s+', ' ', s).strip()
        return s

class SurGEvaluator:
    def __init__(self,device:str = None,survey_path:str = None,corpus_path:str = None,flag_model_path:str = None, judge_model_path:str = None, bertopic_model_path:str = None,bertopic_embedding_model_path:str = None, nli_model_path:str = None, using_openai:bool = True, api_key:str = None):
        import os
        if device != None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        
        self.corpus_dir = corpus_path

        self.using_openai = using_openai
        if using_openai == True:
            assert api_key != None
            #self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
        
        surveys = []
        self.survey_map = {}
        with open(survey_path,'r',encoding='utf-8') as f:
            surveys = json.load(f)
        for s in surveys:
            self.survey_map[int(s['survey_id'])] = s.copy()
        
        corpus = []
        self.corpus_map = {}
        self.title2docid = {}
        with open(corpus_path,'r',encoding='utf-8') as f:
            corpus = json.load(f)
        for c in corpus:
            self.corpus_map[int(c['doc_id'])] = c.copy()
            self.title2docid[normalize_string(c['Title'])] = int(c['doc_id'])
            
        if flag_model_path == None :
            self.flag_model_path = 'BAAI/bge-large-en-v1.5'
        else:
            self.flag_model_path = flag_model_path
            
        # if judge_model_path == None :
        #     self.judge_model_path = None
        # else:
        #     self.judge_model_path = judge_model_path
            
        # self.judge_model = None
        self.flag_model = None
        # if self.judge_model_path != None:
        #     self.judge_model_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_path)
        # else:
        #     self.judge_model_tokenizer = None    
            
        if nli_model_path == None:
            self.nli_model_path = 'cross-encoder/nli-deberta-v3-base'
        else:
            self.nli_model_path = nli_model_path
            
        self.nli_model = None
    
    def compare_section_titles(self, survey_id, psg_node):
        """
        Compara os títulos das seções do artigo gerado pela LLM com os do Golden Truth
        
        Args:
            survey_id: ID do survey no Golden Truth
            psg_node: Nó raiz do artigo gerado (MarkdownNode)
        
        Returns:
            dict com resultados da comparação
        """
        # Extrai títulos do artigo GT
        gt_titles = []
        if survey_id in self.survey_map:
            for section in self.survey_map[survey_id]['structure']:
                if len(section.get('content', '')) >= 10:  # Filtra seções com conteúdo relevante
                    gt_titles.append(section['title'])
        
        # Extrai títulos do artigo gerado pela LLM
        llm_titles = structureFuncs.get_title_list(psg_node)
        
        # Filtra os títulos: remove o primeiro (título do artigo) e seções indesejadas
        excluded_titles = {"Abstract", "References", "Declaration", "Open Access", "Funding", "Acknowledgements", "Author Contributions", "Conflict of Interest"}
        llm_titles = [title for i, title in enumerate(llm_titles) if i > 0 and title not in excluded_titles]
        
        print(f"\n{'='*80}")
        print(f"COMPARAÇÃO DE TÍTULOS DE SEÇÕES - Survey ID: {survey_id}")
        print(f"{'='*80}")
        print(f"Títulos do Golden Truth (GT): {len(gt_titles)}")
        for i, title in enumerate(gt_titles):
            print(f"  {i+1}. {title}")
        print(f"\nTítulos do Artigo LLM: {len(llm_titles)}")
        for i, title in enumerate(llm_titles):
            print(f"  {i+1}. {title}")
        
        # Se não há títulos para comparar, retorna vazio
        if len(gt_titles) == 0 or len(llm_titles) == 0:
            print("\nAviso: Não há títulos suficientes para comparação.")
            return {"gt_titles": gt_titles, "llm_titles": llm_titles, "comparisons": []}
        
        # Gera embeddings usando FlagModel
        if self.flag_model is None:
            self.flag_model = FlagModel(self.flag_model_path, 
                query_instruction_for_retrieval="Generate a representation for this title to calculate the similarity between titles:",
                use_fp16=True)
        
        # Codifica os títulos
        gt_embeddings = self.flag_model.encode(gt_titles)
        llm_embeddings = self.flag_model.encode(llm_titles)
        
        # Normaliza os embeddings para calcular distância coseno
        gt_embeddings_norm = gt_embeddings / np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
        llm_embeddings_norm = llm_embeddings / np.linalg.norm(llm_embeddings, axis=1, keepdims=True)
        
        # Calcula matrix de similaridade coseno
        similarity_matrix = np.dot(gt_embeddings_norm, llm_embeddings_norm.T)
        
        # Converte para distância (1 - similaridade)
        distance_matrix = 1 - similarity_matrix
        
        # Para cada título do GT, encontra o título do LLM mais próximo
        comparisons = []
        for gt_idx, gt_title in enumerate(gt_titles):
            # Encontra o índice do título LLM mais próximo
            min_distance_idx = np.argmin(distance_matrix[gt_idx])
            min_distance = distance_matrix[gt_idx, min_distance_idx]
            llm_title = llm_titles[min_distance_idx]
            
            comparisons.append({
                "gt_title": gt_title,
                "llm_title": llm_title,
                "distance": float(min_distance),
                "similarity": float(similarity_matrix[gt_idx, min_distance_idx])
            })
        
        # Ordena por distância (menor distância = mais similar)
        comparisons_sorted = sorted(comparisons, key=lambda x: x['distance'])
        
        # Filtra apenas comparações com similaridade > 0.8
        threshold = 0.85
        comparisons_above_threshold = [comp for comp in comparisons_sorted if comp['similarity'] > threshold]
        
        # Imprime resultados
        print(f"\n{'-'*120}")
        print("RESULTADOS DA COMPARAÇÃO (Ordenados por menor distância):")
        print(f"Exibindo apenas títulos com similaridade > {threshold}")
        print(f"{'-'*120}")
        print(f"{'Distância':<12} {'Similaridade':<15} {'Título GT':<50} {'Título LLM':<50}")
        print(f"{'-'*120}")
        
        if comparisons_above_threshold:
            for comp in comparisons_above_threshold:
                print(f"{comp['distance']:<12.4f} {comp['similarity']:<15.4f} {comp['gt_title']:<50} {comp['llm_title']:<50}")
        else:
            print("Nenhum título encontrado com similaridade acima do threshold.")
        
        print(f"{'='*80}\n")
        
        # Calcula a métrica de similaridade de títulos
        print("len(comparisons_above_threshold):", len(comparisons_above_threshold))
        print("len(llm_titles):", len(llm_titles))

        subtitle_similarity = len(comparisons_above_threshold) / len(llm_titles) if len(llm_titles) > 0 else 0
        
        return {
            "gt_titles": gt_titles,
            "llm_titles": llm_titles,
            "comparisons": comparisons_sorted,
            "subtitle_similarity": subtitle_similarity
        }
            
    def single_eval(self,survey_id,passage_path,eval_list):
        psg_node = markdownParser.parse_markdown(passage_path)
        refs  = markdownParser.parse_refs(passage_path)
        refid2docid = {}
        
        # Compara títulos das seções se solicitado
        title_comparison_result = None
        if "Compare_Section_Titles" in eval_list or "ALL" in eval_list:
            title_comparison_result = self.compare_section_titles(survey_id, psg_node)

        #print("*****")
        #print(f"psg node: {psg_node}")
        #print(f"refs: {refs}")

        for refid,ref_title in refs.items():

            if normalize_string(ref_title) in self.title2docid:
                ref_docid = self.title2docid[normalize_string(ref_title)]
                refid2docid[refid] = ref_docid
            else:
                refid2docid[refid] = ref_title
        #print(f"ref2docid: {refid2docid}")
        eval_result = {
            "Information_Collection": {
                "Comprehensiveness": {
                    "Coverage": None,
                },
                "Relevance": {
                    "Paper_Level": None,
                    "Section_Level": None,
                    "Sentence_Level": None,
                }
            },
            "Survey_Structure": {
                "Structure_Quality(LLM_as_judge)": None,
                "SH-Recall": None,
                "Subtitle_similarity": None
            },
            "Survey_Content": {
                "Relevance": {
                        "ROUGE-1": None,
                        "ROUGE-2": None,
                        "ROUGE-L": None,
                        "BLEU": None,
                    },
                "Content LLM as a judge": None
            }
        }
        
        if "ROUGE-BLEU" in eval_list or "ALL" in eval_list:
            r1,r2,rl,bleu = rougeBleuFuncs.eval_rougeBleu(self.survey_map[survey_id],psg_node)
            eval_result["Survey_Content"]["Relevance"]["ROUGE-1"] = r1
            eval_result["Survey_Content"]["Relevance"]["ROUGE-2"] = r2
            eval_result["Survey_Content"]["Relevance"]["ROUGE-L"] = rl
            eval_result["Survey_Content"]["Relevance"]["BLEU"] = bleu
        
        if "subtitle_similarity" in eval_list or "ALL" in eval_list:
            if title_comparison_result is not None:
                eval_result["Survey_Structure"]["subtitle_similarity"] = float(title_comparison_result["subtitle_similarity"])
        
        if "SH-Recall" in eval_list or "ALL" in eval_list:
            if self.flag_model == None:
                self.flag_model = FlagModel(self.flag_model_path, 
                    query_instruction_for_retrieval="Generate a representation for this title to calculate the similarity between titles:",
                        use_fp16=True)  
            
            sh_recall = structureFuncs.eval_SHRecall(self.survey_map[survey_id],psg_node,self.flag_model)
            eval_result["Survey_Structure"]["SH-Recall"] = float(sh_recall)
        
        if "Structure_Quality" in eval_list or "ALL" in eval_list:
            # if self.judge_model == None and self.using_openai == False:
            #     self.judge_model = AutoModelForCausalLM.from_pretrained(
            #         self.judge_model_path,
            #         torch_dtype= torch.float16,
            #         device_map="auto"
            #     )
            if self.using_openai == True:
                struct_quality = structureFuncs.eval_structure_quality_client(self.survey_map[survey_id],psg_node,self.client)
            else:
                pass 
                # struct_quality = structureFuncs.eval_structure_quality(self.survey_map[survey_id],psg_node,self.judge_model,self.judge_model_tokenizer)
            eval_result["Survey_Structure"]["Structure_Quality(LLM_as_judge)"] = struct_quality 
        
        if "Coverage" in eval_list or "ALL" in eval_list:
            coverage = informationFuncs.eval_coverage(self.survey_map[survey_id]['all_cites'],refid2docid)
            eval_result["Information_Collection"]["Comprehensiveness"]["Coverage"] = coverage
            
        if "Relevance-Paper" in eval_list or "ALL" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            refcontent = {}
            for k,v in refid2docid.items():
                sen_1 = None
                sen_paper = None 
                #print(f"K: {k}, V: {v}")
                if isinstance(v,int):
                    tmp_1 = self.corpus_map[v]['Title']
                    tmp_2 = self.corpus_map[v]['Abstract']
                    tmp_title = self.survey_map[survey_id]['survey_title']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: '{tmp_2}'"
                    sen_paper = f"The paper titled '{tmp_1}' with the given abstract could be cited in the paper: '{tmp_title}'."
                    refcontent[k] = (sen_1,sen_paper)
                else:
                    tmp_title = self.survey_map[survey_id]['survey_title']
                    # sen_1 = refcontent[k] = f"There is a paper. Title: '{v}'. The title '{v}' describes the content of the paper."
                    # sen_paper = f"The paper titled '{v}' could be cited in the paper: '{tmp_title}'."
                    sen_1 = "[NOTEXIST]"
                    sen_paper = "[NOTEXIST]"
                    refcontent[k] = (sen_1,sen_paper)
            paper_relevance = None
            if len(refid2docid) > 0:
                paper_relevance = informationFuncs.eval_relevance_paper(self.survey_map[survey_id],refid2docid,refcontent,self.nli_model)
                print(f"Paper Relevance: {paper_relevance}")
            else:
                paper_relevance = 0
            eval_result["Information_Collection"]["Relevance"]["Paper_Level"] = paper_relevance
        
        if ("Relevance-Section" in eval_list and "Relevance-Sentence" in eval_list) or "ALL" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_subtitle = []
            nli_pairs_sentence = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                sen_sentence = None
                sen_section = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                    sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'."
                    # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                    sen_1 = "[NOTEXIST]"
                    sen_section = "[NOTEXIST]"
                    sen_sentence = "[NOTEXIST]"
                nli_pairs_sentence.append((sen_1,sen_sentence))
                nli_pairs_subtitle.append((sen_1,sen_section))
            section_relevance = None
            sentence_relevance = None
            if len(extracted_cites) > 0:    
                section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
                sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            else:
                sentence_relevance = 0
                section_relevance = 0
            eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
            eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
        elif "Relevance-Section" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_subtitle = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_section = f"The paper titled '{tmp_1}' with the given abstract is relevant to the section: '{subtitle}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_section = f"The paper titled '{docid}' is relevant to the section: '{subtitle}'." 
                    sen_1 = "[NOTEXIST]"
                    sen_section = "[NOTEXIST]"

                nli_pairs_subtitle.append((sen_1,sen_section))
            section_relevance = None
            if len(extracted_cites) > 0:    
                section_relevance = informationFuncs.eval_relevance_section(nli_pairs_subtitle,self.nli_model)
            else:
                section_relevance = 0
            
            eval_result["Information_Collection"]["Relevance"]["Section_Level"] = section_relevance
        elif "Relevance-Sentence" in eval_list:
            if self.nli_model == None:
                self.nli_model = CrossEncoder(self.nli_model_path)
            extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
            nli_pairs_sentence = []
            for ref_num,subtitle,sentence in extracted_cites:
                if ref_num not in refid2docid:
                    docid = "This is an irrelevant paper."
                else:
                    docid = refid2docid[ref_num]
                sen_1 = None
                if isinstance(docid,int):
                    tmp_1 = self.corpus_map[docid]['Title']
                    tmp_2 = self.corpus_map[docid]['Abstract']
                    sen_1 = f"There is a paper. Title: '{tmp_1}'. Abstract: {tmp_2}"
                    sen_sentence = f"The paper titled '{tmp_1}' with the given abstract could be cited in the sentence: '{sentence}'."
                else:
                    # The title
                    
                    # sen_1 = f"There is a paper. Title: '{docid}'. The title '{docid}' describes the content of the paper."
                    # sen_sentence = f"The paper titled '{docid}' could be cited in the sentence: '{sentence}'."
                    sen_1 = "[NOTEXIST]"
                    sen_sentence = "[NOTEXIST]"
                nli_pairs_sentence.append((sen_1,sen_sentence))
            
            sentence_relevance = None
            if len(extracted_cites) > 0:    
                sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            else:
                sentence_relevance = 0
            sentence_relevance = informationFuncs.eval_relevance_sentence(nli_pairs_sentence,self.nli_model)
            eval_result["Information_Collection"]["Relevance"]["Sentence_Level"] = sentence_relevance
            
        if "Content" in eval_list or "ALL" in eval_list:
            if self.using_openai == True:
                content_llm_judge = informationFuncs.eval_content_client(psg_node,self.client)
            else:
                pass 
            eval_result["Survey_Content"]["Content LLM as a judge"] = content_llm_judge   
        
        return eval_result
            
        
    def eval_all(self,passage_dir,eval_list,save_path = None):
        print(f"Starting evaluation with passage_dir: {passage_dir}, eval_list: {eval_list}, save_path: {save_path}")
        if save_path != None :
            with open (save_path,"w",encoding='utf-8') as f:
                f.write('')
                
        survey_ids = [
            d for d in os.listdir(passage_dir)
            if os.path.isdir(os.path.join(passage_dir, d))
        ]
        print("survey_ids:", survey_ids)

        length = len(survey_ids)
        
        eval_result = {
            "Information_Collection": {
                "Comprehensiveness": {
                    "Coverage": None,
                },
                "Relevance": {
                    "Paper_Level": None,
                    "Section_Level": None,
                    "Sentence_Level": None,
                }
            },
            "Survey_Structure": {
                "Structure_Quality(LLM_as_judge)": None,
                "SH-Recall": None,
                "subtitle_similarity": None
            },
            "Survey_Content": {
                "Relevance": {
                        "ROUGE-1": None,
                        "ROUGE-2": None,
                        "ROUGE-L": None,
                        "BLEU": None,
                    },
                "Content LLM as a judge": None,
            }
        }

        
        for survey_id in tqdm(survey_ids):
            survey_dir = os.path.join(passage_dir,survey_id)
            
            psg_files = os.listdir(survey_dir)
            tmp_res = None
            
            if len(psg_files) == 1:
                psg_path = os.path.join(survey_dir,psg_files[0])
                tmp_res = self.single_eval(int(survey_id),psg_path,eval_list)
            else:
                for psg_file in psg_files:
                    if psg_file.endswith('.md'):
                        psg_path = os.path.join(survey_dir,psg_file)
                        tmp_res = self.single_eval(int(survey_id),psg_path,eval_list)
                        break
            # print(tmp_res)

            
            for k1,v1 in tmp_res.items():
                for k2,v2 in v1.items():
                    if isinstance(v2,dict):
                        for k3,v3 in v2.items():    
                            if v3 != None:
                                if eval_result[k1][k2][k3] == None:
                                    eval_result[k1][k2][k3] = v3/length
                                else:
                                    eval_result[k1][k2][k3] += v3/length
                    else:
                        if v2 != None:
                            if eval_result[k1][k2] == None:
                                eval_result[k1][k2] = v2/length
                            else:
                                eval_result[k1][k2] += v2/length
                                
            if save_path != None:
                tmp_res['survey_id'] = survey_id
                with open (save_path,"a",encoding='utf-8') as f:
                    json.dump(tmp_res,f,ensure_ascii=False,indent=4)
                    f.write('\n')
                print(f"Result of {survey_id}")
                print(tmp_res)
            else:
                print(f"Result of {survey_id}")
                print(tmp_res)    
        
        if save_path != None:
            eval_result['survey_id'] = "Average"
            with open (save_path,"a",encoding='utf-8') as f:
                f.write('\n')
                json.dump(eval_result,f,ensure_ascii=False,indent=4)
                print("Total Result:")
                print(eval_result)
        else:
            print("Total Result:")
            print(eval_result)
            
        return eval_result
    
    def compare_sections_detailed(self, llm_md_path: str, gt_md_path: str, threshold: float = 0.85):
        """
        Compara seções/subseções entre artigo LLM e GT com análise detalhada de conteúdo.
        
        Args:
            llm_md_path: Caminho do arquivo markdown gerado pela LLM
            gt_md_path: Caminho do arquivo markdown do Ground Truth
            threshold: Threshold de similaridade para filtrar matches (default 0.85)
            
        Returns:
            list de dicts com comparações detalhadas
        """
        import re
        
        # Parse dos dois arquivos markdown
        llm_node = markdownParser.parse_markdown(llm_md_path)
        gt_node = markdownParser.parse_markdown(gt_md_path)
        
        # Extrai listas de títulos
        llm_titles = structureFuncs.get_title_list(llm_node)
        gt_titles = structureFuncs.get_title_list(gt_node)
        
        print(f"\n{'='*100}")
        print(f"ANÁLISE DETALHADA DE SEÇÕES COM CONTEÚDO")
        print(f"{'='*100}")
        print(f"Títulos LLM: {len(llm_titles)}")
        print(f"Títulos GT: {len(gt_titles)}")
        
        if len(gt_titles) == 0 or len(llm_titles) == 0:
            print("Aviso: Não há títulos suficientes para comparação.")
            return []
        
        # Gera embeddings
        if self.flag_model is None:
            self.flag_model = FlagModel(self.flag_model_path,
                query_instruction_for_retrieval="Generate a representation for this title:",
                use_fp16=True)
        
        gt_embeddings = self.flag_model.encode(gt_titles)
        llm_embeddings = self.flag_model.encode(llm_titles)
        
        # Normaliza embeddings
        gt_embeddings_norm = gt_embeddings / np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
        llm_embeddings_norm = llm_embeddings / np.linalg.norm(llm_embeddings, axis=1, keepdims=True)
        
        # Calcula matriz de similaridade
        similarity_matrix = np.dot(gt_embeddings_norm, llm_embeddings_norm.T)
        distance_matrix = 1 - similarity_matrix
        
        # Encontra aplicar matching greedy e filtra por threshold
        comparison_results = []
        
        for gt_idx, gt_title in enumerate(gt_titles):
            min_distance_idx = np.argmin(distance_matrix[gt_idx])
            min_distance = distance_matrix[gt_idx, min_distance_idx]
            similarity = similarity_matrix[gt_idx, min_distance_idx]
            llm_title = llm_titles[min_distance_idx]
            
            # Filtra por threshold
            if similarity >= threshold:
                # Extrai nó correspondente em GT
                gt_section_node = self._find_node_by_title(gt_node, gt_title)
                # Extrai nó correspondente em LLM
                llm_section_node = self._find_node_by_title(llm_node, llm_title)
                
                # Extrai textos
                text_gt = structureFuncs.extract_section_text(gt_section_node) if gt_section_node else ""
                text_llm = structureFuncs.extract_section_text(llm_section_node) if llm_section_node else ""
                
                # Calcula métricas se houver texto
                metrics = {'bertscore_f1': 0.0, 'rouge_l': 0.0}
                if len(text_gt) > 50 and len(text_llm) > 50:
                    metrics = structureFuncs.calculate_content_metrics(text_llm, text_gt)
                
                comparison_results.append({
                    'section_llm_name': llm_title,
                    'section_gt_name': gt_title,
                    'distance': float(min_distance),
                    'similarity': float(similarity),
                    'bertscore_f1': float(metrics['bertscore_f1']),
                    'rouge_l': float(metrics['rouge_l']),
                    'text_llm': text_llm,
                    'text_gt': text_gt
                })
        
        print(f"Matches encontrados com similaridade >= {threshold}: {len(comparison_results)}")
        print(f"{'='*100}\n")
        
        return comparison_results
    
    def _find_node_by_title(self, root_node: 'markdownParser.MarkdownNode', title: str) -> 'MarkdownNode':
        """
        Encontra um nó na árvore MarkdownNode pelo título.
        
        Args:
            root_node: Nó raiz para começar a busca
            title: Título a buscar
            
        Returns:
            MarkdownNode encontrado ou None
        """
        if root_node.title == title:
            return root_node
        
        for child in root_node.children:
            result = self._find_node_by_title(child, title)
            if result:
                return result
        
        return None
    
    def export_sections_comparison(self, comparison_data: list, output_path: str) -> None:
        """
        Exporta comparação detalhada de seções em formato de tabela em arquivo texto.
        
        Args:
            comparison_data: Lista de dicts com comparações
            output_path: Caminho do arquivo de saída
        """
        import os
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Escreve cabeçalho
            f.write("="*200 + "\n")
            f.write("COMPARAÇÃO DETALHADA DE SEÇÕES SIMILARES (Similaridade >= 0.85)\n")
            f.write("="*200 + "\n\n")
            
            if not comparison_data:
                f.write("Nenhuma seção encontrada com similaridade acima do threshold.\n")
                return
            
            # Cabeçalho da tabela
            headers = ["Section LLM", "Section GT", "Distance", "Similarity", "BERTScore", "ROUGE-L"]
            col_widths = [80, 80, 15, 15, 15, 15]
            
            header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
            f.write(header_line + "\n")
            f.write("-" * len(header_line) + "\n")
            
            # Escreve linhas de dados
            for item in comparison_data:
                row = [
                    item['section_llm_name'],
                    item['section_gt_name'],
                    f"{item['distance']:.4f}",
                    f"{item['similarity']:.4f}",
                    f"{item['bertscore_f1']:.4f}",
                    f"{item['rouge_l']:.4f}"
                ]
                formatted_row = " | ".join(f"{str(r):<{col_widths[i]}}" for i, r in enumerate(row))
                f.write(formatted_row + "\n")
            
            # Escreve textos das seções
            f.write("\n" + "="*200 + "\n")
            f.write("TEXTOS COMPLETOS DAS SEÇÕES\n")
            f.write("="*200 + "\n\n")
            
            for idx, item in enumerate(comparison_data, 1):
                f.write(f"\n{'-'*200}\n")
                f.write(f"SEÇÃO {idx}: {item['section_gt_name']} (GT) ↔ {item['section_llm_name']} (LLM)\n")
                f.write(f"Similaridade: {item['similarity']:.4f} | BERTScore: {item['bertscore_f1']:.4f} | ROUGE-L: {item['rouge_l']:.4f}\n")
                f.write(f"{'-'*200}\n")
                
                f.write(f"\n[TEXTO GT]:\n{item['text_gt']}\n")
                f.write(f"\n[TEXTO LLM]:\n{item['text_llm']}\n")
        
        print(f"Comparação exportada para: {output_path}")
