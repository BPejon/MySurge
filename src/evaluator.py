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
    
    def extract_sections_hierarchical_from_llm(self, markdown_node, level=0, parent_path=None):
        """
        Extrai seções e subsecções da árvore MarkdownNode de forma hierárquica.
        Retorna conteúdo DIRETO de cada nível (sem incluir subsecções aninhadas).
        
        Args:
            markdown_node: MarkdownNode raiz
            level: Nível hierárquico (0 = raiz, 1 = seção, 2 = subsecção, etc)
            parent_path: Lista de títulos pais até este nó
            
        Returns:
            list: Lista de dicts com estrutura:
                {
                    'title': str,
                    'level': int,
                    'content': str (conteúdo direto),
                    'path_titles': list (caminho hierárquico),
                    'node': MarkdownNode (referência ao nó)
                }
        """
        sections = []
        
        if parent_path is None:
            parent_path = []
        
        if markdown_node is None:
            return sections
        
        # Pula o nó raiz (título do artigo)
        if level > 0:
            # Extrai conteúdo direto deste nó (sem recursão para filhos)
            content_text = ""
            if hasattr(markdown_node, 'content') and markdown_node.content:
                content_text = "\n".join(markdown_node.content).strip()
            
            # Só adiciona se houver conteúdo ou título relevante
            if content_text and len(content_text) >= 10:
                current_path = parent_path + [markdown_node.title]
                
                sections.append({
                    'title': markdown_node.title,
                    'level': level,
                    'content': content_text,
                    'path_titles': current_path,
                    'node': markdown_node
                })
        
        # Processa filhos recursivamente
        if hasattr(markdown_node, 'children') and markdown_node.children:
            new_parent_path = parent_path + ([markdown_node.title] if level > 0 else [])
            
            for child in markdown_node.children:
                child_sections = self.extract_sections_hierarchical_from_llm(
                    child, 
                    level=level + 1, 
                    parent_path=new_parent_path
                )
                sections.extend(child_sections)
        
        return sections
    
    def extract_sections_hierarchical_from_gt(self, survey_id):
        """
        Extrai seções e subsecções do Ground Truth de forma hierárquica.
        Retorna conteúdo DIRETO de cada nível (sem incluir subsecções aninjadas).
        
        Args:
            survey_id: ID do survey no survey_map
            
        Returns:
            list: Lista de dicts com estrutura:
                {
                    'title': str,
                    'level': int,
                    'content': str (conteúdo direto),
                    'path_titles': list (caminho hierárquico),
                    'section_dict': dict (referência à seção original)
                }
        """
        sections = []
        
        if survey_id not in self.survey_map:
            return sections
        
        survey = self.survey_map[survey_id]
        
        # Primeiro: construir mapa global de ID -> seção em UMA única passagem
        id_to_section = {}
        
        def build_id_map(section_list):
            """Constrói mapa de IDs em uma única passagem sem recursão"""
            for section in section_list:
                section_id = section.get('id')
                if section_id:
                    id_to_section[section_id] = section
        
        if 'structure' in survey:
            build_id_map(survey['structure'])
        
        # Segundo: processar seções com limite de profundidade
        MAX_DEPTH = 10  # Evita recursão infinita
        
        def process_sections_recursive(section_list, level, parent_path, depth=0):
            """Processa recursivamente lista de seções com limite de profundidade"""
            if not section_list or depth > MAX_DEPTH:
                return
            
            for section in section_list:
                # Extrai conteúdo direto
                content = section.get('content', '').strip()
                current_path = parent_path + [section['title']]
                
                # Só adiciona se houver conteúdo relevante
                if content and len(content) >= 10:
                    sections.append({
                        'title': section['title'],
                        'level': level,
                        'content': content,
                        'path_titles': current_path,
                        'section_dict': section
                    })
                
                # Processa subsecções usando o mapa de IDs
                if 'subsections' in section and section['subsections'] and level < 5:
                    subsection_objs = []
                    for subsec_id in section['subsections']:
                        if subsec_id in id_to_section:
                            subsection_objs.append(id_to_section[subsec_id])
                    
                    if subsection_objs:
                        process_sections_recursive(subsection_objs, level + 1, current_path, depth + 1)
        
        # Inicia processo recursivo
        if 'structure' in survey:
            process_sections_recursive(survey['structure'], 1, [], 0)
        
        return sections
    
    def compare_section_content_hierarchical(self, survey_id, psg_node, similarity_threshold=0.75):
        """
        Compara trechos de conteúdo de seções e subsecções entre GT e LLM.
        Calcula matriz de similaridade coseno e identifica pares similares.
        
        Args:
            survey_id: ID do survey
            psg_node: Raiz MarkdownNode do artigo LLM
            similarity_threshold: Threshold mínimo de similaridade (default 0.75)
            
        Returns:
            dict com:
                - gt_sections: Lista de seções GT extraídas
                - llm_sections: Lista de seções LLM extraídas
                - similarity_matrix: Matriz numpy de similaridades
                - similar_pairs: Lista de pares similares ordenados
                - comparison_df: DataFrame com resultados detalhados
        """
        import pandas as pd
        
        print(f"\n{'='*100}")
        print(f"COMPARAÇÃO DE TRECHOS DE CONTEÚDO - Survey ID: {survey_id}")
        print(f"{'='*100}")
        
        # Extrai seções hierarquicamente
        print("Extraindo seções do Ground Truth...")
        gt_sections = self.extract_sections_hierarchical_from_gt(survey_id)
        print(f"  Total de seções/subsecções GT: {len(gt_sections)}")
        
        print("Extraindo seções do artigo LLM...")
        llm_sections = self.extract_sections_hierarchical_from_llm(psg_node)
        print(f"  Total de seções/subsecções LLM: {len(llm_sections)}")
        
        if len(gt_sections) == 0 or len(llm_sections) == 0:
            print("Aviso: Não há seções suficientes para comparação.")
            return {
                "gt_sections": gt_sections,
                "llm_sections": llm_sections,
                "similarity_matrix": np.array([]),
                "similar_pairs": [],
                "comparison_df": pd.DataFrame()
            }
        
        # Gera embeddings usando FlagModel
        print("\nGerando embeddings dos trechos...")
        if self.flag_model is None:
            self.flag_model = FlagModel(self.flag_model_path,
                query_instruction_for_retrieval="Generate a representation for this text to calculate similarity:",
                use_fp16=True)
        
        # Extrai conteúdos e codifica
        gt_contents = [sec['content'] for sec in gt_sections]
        llm_contents = [sec['content'] for sec in llm_sections]
        
        gt_embeddings = self.flag_model.encode(gt_contents)
        llm_embeddings = self.flag_model.encode(llm_contents)
        
        # Normaliza embeddings para similaridade coseno
        gt_embeddings_norm = gt_embeddings / np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
        llm_embeddings_norm = llm_embeddings / np.linalg.norm(llm_embeddings, axis=1, keepdims=True)
        
        # Calcula matriz de similaridade coseno
        similarity_matrix = np.dot(gt_embeddings_norm, llm_embeddings_norm.T)
        
        print(f"Matriz de similaridade gerada: {similarity_matrix.shape}")
        
        # Identifica pares similares
        similar_pairs = []
        
        for i, gt_sec in enumerate(gt_sections):
            for j, llm_sec in enumerate(llm_sections):
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    similar_pairs.append({
                        'gt_title': gt_sec['title'],
                        'gt_level': gt_sec['level'],
                        'gt_content': gt_sec['content'],
                        'gt_path': ' > '.join(gt_sec['path_titles']),
                        'llm_title': llm_sec['title'],
                        'llm_level': llm_sec['level'],
                        'llm_content': llm_sec['content'],
                        'llm_path': ' > '.join(llm_sec['path_titles']),
                        'similarity': similarity,
                        'gt_idx': i,
                        'llm_idx': j
                    })
        
        # Ordena por similaridade decrescente
        similar_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
        
        print(f"\nPares similares encontrados (threshold >= {similarity_threshold}): {len(similar_pairs)}")
        
        # Cria DataFrame para exibição
        comparison_data = []
        for pair in similar_pairs:
            comparison_data.append({
                'GT_Título': pair['gt_title'],
                'LLM_Título': pair['llm_title'],
                'Nível_GT': f"Nível {pair['gt_level']}",
                'Nível_LLM': f"Nível {pair['llm_level']}",
                'Similaridade': round(pair['similarity'], 4),
                'Caminho_GT': pair['gt_path'],
                'Caminho_LLM': pair['llm_path'],
                'Tamanho_GT': len(pair['gt_content'].split()),
                'Tamanho_LLM': len(pair['llm_content'].split())
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return {
            "gt_sections": gt_sections,
            "llm_sections": llm_sections,
            "similarity_matrix": similarity_matrix,
            "similar_pairs": similar_pairs,
            "comparison_df": comparison_df
        }
    
    def print_similar_pairs_table(self, comparison_result):
        """
        Imprime tabela formatada dos pares de seções mais similares.
        
        Args:
            comparison_result: Dict retornado por compare_section_content_hierarchical()
        """
        comparison_df = comparison_result.get('comparison_df')
        
        if comparison_df is None or comparison_df.empty:
            print("Tabela vazia. Nenhum par similar encontrado.")
            return
        
        print("\n" + "="*180)
        print("SEÇÕES MAIS SIMILARES - COMPARAÇÃO ENTRE GT E LLM")
        print("="*180)
        
        # Cabeçalho
        print(f"{'GT Título':<35} {'LLM Título':<35} {'Nível GT':<12} {'Nível LLM':<12} {'Similaridade':<14} {'Tamanho GT':<12} {'Tamanho LLM':<12}")
        print("-"*180)
        
        # Dados
        for idx, row in comparison_df.iterrows():
            gt_title = str(row['GT_Título'])[:33]
            llm_title = str(row['LLM_Título'])[:33]
            
            print(f"{gt_title:<35} {llm_title:<35} {str(row['Nível_GT']):<12} {str(row['Nível_LLM']):<12} "
                  f"{row['Similaridade']:<14.4f} {row['Tamanho_GT']:<12} {row['Tamanho_LLM']:<12}")
        
        print("="*180)
        print(f"Total de pares similares: {len(comparison_df)}")
        print(f"Similaridade Média: {comparison_df['Similaridade'].mean():.4f}")
        print(f"Similaridade Máxima: {comparison_df['Similaridade'].max():.4f}")
        print(f"Similaridade Mínima: {comparison_df['Similaridade'].min():.4f}")
        print()
    
    def extract_section_text_from_gt(self, section_dict):
        """
        Extrai o texto de conteúdo de uma seção do Ground Truth (estrutura JSON).
        
        Args:
            section_dict (dict): Dicionário da seção com chave 'content'
            
        Returns:
            str: Texto da seção ou NaN se vazio/inválido
        """
        import numpy as np
        
        if section_dict is None or not isinstance(section_dict, dict):
            return np.nan
        
        content = section_dict.get('content', '')
        
        if not content or not str(content).strip():
            return np.nan
        
        return str(content).strip()
    
    def find_section_by_title(self, title, section_list):
        """
        Encontra uma seção pelo título na lista de seções.
        
        Args:
            title (str): Título da seção a buscar
            section_list (list): Lista de seções (dicts com 'title' e 'content')
            
        Returns:
            dict: Seção encontrada ou None
        """
        for section in section_list:
            if section.get('title') == title:
                return section
        return None
    
    def find_markdown_section_by_title(self, title, markdown_node):
        """
        Encontra um nó MarkdownNode pelo título na árvore de markdown.
        Busca em largura (BFS) para evitar recursão profunda.
        
        Args:
            title (str): Título a buscar
            markdown_node (MarkdownNode): Nó raiz para começar busca
            
        Returns:
            MarkdownNode: Nó encontrado ou None
        """
        if markdown_node is None:
            return None
        
        from collections import deque
        
        queue = deque([markdown_node])
        
        while queue:
            node = queue.popleft()
            
            if hasattr(node, 'title') and node.title == title:
                return node
            
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    queue.append(child)
        
        return None
    
    def generate_comparison_table(self, comparison_result, psg_node):
        """
        Gera tabela de comparação entre seções GT e LLM com BERTScore.
        
        Args:
            comparison_result (dict): Resultado de compare_section_titles()
            psg_node (MarkdownNode): Raiz do artigo LLM parseado
            
        Returns:
            pandas.DataFrame: Tabela com colunas de comparação e BERTScore
        """
        import pandas as pd
        from rougeBleuFuncs import calculate_bertscore_for_sections
        from structureFuncs import extract_section_text_from_markdown
        import numpy as np
        
        comparisons = comparison_result.get('comparisons', [])
        
        if not comparisons:
            print("Nenhuma comparação para gerar tabela.")
            return pd.DataFrame()
        
        # Dados para a tabela
        table_data = []
        
        for comp in comparisons:
            gt_title = comp.get('gt_title', '')
            llm_title = comp.get('llm_title', '')
            distance = comp.get('distance', np.nan)
            similarity = comp.get('similarity', np.nan)
            
            # Extrair textos das seções
            text_gt = np.nan
            text_llm = np.nan
            
            # Extrair texto GT
            if hasattr(self, 'survey_map'):
                for survey_id in self.survey_map:
                    survey = self.survey_map[survey_id]
                    if 'structure' in survey:
                        gt_section = self.find_section_by_title(gt_title, survey['structure'])
                        if gt_section:
                            text_gt = self.extract_section_text_from_gt(gt_section)
                            break
            
            # Extrair texto LLM
            llm_section = self.find_markdown_section_by_title(llm_title, psg_node)
            if llm_section:
                text_llm = extract_section_text_from_markdown(llm_section)
                if not text_llm or not text_llm.strip():
                    text_llm = np.nan
            
            # Calcular BERTScore
            bertscore_dict = calculate_bertscore_for_sections(
                str(text_llm) if not pd.isna(text_llm) else "",
                str(text_gt) if not pd.isna(text_gt) else ""
            )
            
            # Montar linha da tabela
            table_data.append({
                'seção_llm': llm_title[:40] + '...' if len(str(llm_title)) > 40 else llm_title,
                'seção_gt': gt_title[:40] + '...' if len(str(gt_title)) > 40 else gt_title,
                'distância': round(distance, 4) if not pd.isna(distance) else np.nan,
                'similaridade': round(similarity, 4) if not pd.isna(similarity) else np.nan,
                'bertscore_f1': round(bertscore_dict.get('f1', np.nan), 4),
                'texto_llm': (str(text_llm)[:100] + '...') if not pd.isna(text_llm) and len(str(text_llm)) > 100 else (str(text_llm) if not pd.isna(text_llm) else 'N/A'),
                'texto_gt': (str(text_gt)[:100] + '...') if not pd.isna(text_gt) and len(str(text_gt)) > 100 else (str(text_gt) if not pd.isna(text_gt) else 'N/A')
            })
        
        df = pd.DataFrame(table_data)
        return df
    
    def print_comparison_table(self, df):
        """
        Imprime a tabela de comparação em formato legível.
        
        Args:
            df (pandas.DataFrame): DataFrame com dados de comparação
        """
        if df.empty:
            print("Tabela vazia.")
            return
        
        print("\n" + "="*200)
        print("ANÁLISE COMPARATIVA: BERTSCORE E SIMILARIDADE DE SEÇÕES")
        print("="*200)
        
        # Imprime cabeçalho
        print(f"{'Seção LLM':<45} {'Seção GT':<45} {'Dist':<8} {'Sim':<8} {'F1':<8} {'Texto LLM':<55} {'Texto GT':<55}")
        print("-"*200)
        
        # Imprime cada linha
        for idx, row in df.iterrows():
            print(f"{str(row['seção_llm']):<45} {str(row['seção_gt']):<45} "
                  f"{str(row['distância']):<8} {str(row['similaridade']):<8} "
                  f"{str(row['bertscore_f1']):<8} {str(row['texto_llm']):<55} {str(row['texto_gt']):<55}")
        
        print("="*200)
        print(f"Total de seções comparadas: {len(df)}")
        print(f"BERTScore F1 Médio: {df['bertscore_f1'].mean():.4f}")
        print(f"Similaridade Média: {df['similaridade'].mean():.4f}")
        print()
            
    def single_eval(self,survey_id,passage_path,eval_list):
        psg_node = markdownParser.parse_markdown(passage_path)
        refs  = markdownParser.parse_refs(passage_path)
        refid2docid = {}
        
        # Compara títulos das seções se solicitado
        title_comparison_result = None
        if "Compare_Section_Titles" in eval_list or "ALL" in eval_list:
            title_comparison_result = self.compare_section_titles(survey_id, psg_node)
            
            # Gera análise com BERTScore se comparação foi feita
            try:
                comparison_df = self.generate_comparison_table(title_comparison_result, psg_node)
                if not comparison_df.empty:
                    self.print_comparison_table(comparison_df)
            except Exception as e:
                print(f"Erro ao gerar tabela de comparação com BERTScore: {e}")

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